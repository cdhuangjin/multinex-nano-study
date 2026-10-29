"""Matched research training runner, not a claim of bitwise official retraining.

Uses unchanged official architecture, HybridLoss, and LR scheduler. Data draws
are step-indexed so gate construction and checkpoint resume cannot alter them.
"""
import argparse
import csv
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import types

import cv2
import numpy as np
import torch
import yaml



HERE = Path(__file__).resolve().parent
ROOT = HERE
sys.path.insert(0, str(ROOT))
from official_adapter import CODE, load_file, official_metrics


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


@lru_cache(maxsize=64)
def rgb(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    if x is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(x, cv2.COLOR_BGR2RGB)


def batch(config, split, step):
    opts = config['datasets']['train']
    rng = np.random.default_rng(np.random.SeedSequence([config['manual_seed'], step]))
    chosen = rng.choice(split['train'], opts['batch_size_per_gpu'], replace=True).tolist()
    lows, highs, records = [], [], []
    for name in chosen:
        low, high = [rgb(split['files'][name][k]['path']) for k in ['low','high']]
        h,w = low.shape[:2]
        size = opts['gt_size']
        top, left = int(rng.integers(h-size+1)), int(rng.integers(w-size+1))
        aug = int(rng.integers(8))
        pair = []
        for a in [low,high]:
            a = np.rot90(a[top:top+size,left:left+size], aug//2)
            if aug%2:
                a = np.flipud(a)
            pair.append(a.copy().transpose(2,0,1).astype(np.float32)/255)
        lows.append(pair[0]); highs.append(pair[1])
        records.append([name,top,left,aug])
    x,y = torch.from_numpy(np.stack(lows)).cuda(), torch.from_numpy(np.stack(highs)).cuda()
    mix = int(rng.integers(2)) == 0
    if mix:
        lam = float(rng.beta(1.2,1.2))
        perm = rng.permutation(len(chosen))
        x,y = lam*x+(1-lam)*x[perm], lam*y+(1-lam)*y[perm]
        records.append(['mix',lam,perm.tolist()])
    return x,y,hashlib.sha256(json.dumps(records).encode()).hexdigest()


def criterion(config):
    cache = Path(config['experiment']['torch_home'])/'hub/checkpoints/vgg19-dcbb9e9d.pth'
    if not cache.is_file():
        raise FileNotFoundError(f'Complete official VGG19 weights first: {cache}; run prepare_vgg.py')
    digest = sha(cache)
    if not digest.startswith('dcbb9e9d'):
        raise ValueError('VGG19 checksum prefix mismatch')
    os.environ['TORCH_HOME'] = config['experiment']['torch_home']
    official_metrics()
    for name,directory in [('basicsr.models','basicsr/models'), ('basicsr.models.losses','basicsr/models/losses')]:
        shell = types.ModuleType(name); shell.__path__ = [str(CODE/directory)]; sys.modules[name] = shell
    load_file('basicsr.models.losses.loss_util','basicsr/models/losses/loss_util.py')
    module = load_file('basicsr.models.losses.losses','basicsr/models/losses/losses.py')
    opts = dict(config['train']['pixel_opt']); assert opts.pop('type') == 'HybridLoss'
    loss = module.HybridLoss(**opts).cuda()
    # Initialize the frozen official feature net before timing, without changing its forward.
    loss.perceptual_loss_model = module.VGGPerceptualLoss().cuda().eval()
    return loss, digest


def validate(model, split, out, step):
    utils = official_metrics()
    rows = []
    model.eval()
    start = time.perf_counter()
    with torch.inference_mode():
        for name in split['dev']:
            low, high = [rgb(split['files'][name][k]['path']).astype(np.float32)/255 for k in ['low','high']]
            x = torch.from_numpy(low.transpose(2,0,1).copy()).unsqueeze(0).cuda()
            pred = model(x).clamp(0,1)[0].permute(1,2,0).cpu().numpy()
            rows.append({'image':name,'group':split['group_map'][name],
                         'psnr':float(utils.PSNR(pred,high,False)),
                         'ssim':float(utils.calculate_ssim(pred,high,border=0,gtmean=False))})
    with (out/f'dev-{step:06}.csv').open('w',newline='',encoding='utf-8') as f:
        writer = csv.DictWriter(f,fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    model.train()
    return {key:float(np.mean([r[key] for r in rows])) for key in ['psnr','ssim']} | {'seconds':time.perf_counter()-start}

