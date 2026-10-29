"""Evaluate an existing checkpoint on paired image folders (CPU by default)."""
import argparse, csv, json
from pathlib import Path
import numpy as np
import torch
from variant_model import build
from train_controlled import rgb
from official_adapter import official_metrics
p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); p.add_argument('--arm',choices=['baseline','rep','A','B'],required=True)
p.add_argument('--low',type=Path,required=True); p.add_argument('--high',type=Path,required=True)
p.add_argument('--names',type=Path,help='Optional newline-separated image names, e.g. dev.txt')
p.add_argument('--output',type=Path,default=Path('evaluation.csv')); a=p.parse_args()
torch.set_num_threads(4); model=build(a.arm,110).eval()
state=torch.load(a.checkpoint,map_location='cpu',weights_only=True); model.load_state_dict(state['params'])
u=official_metrics(); rows=[]
names=a.names.read_text().splitlines() if a.names else sorted(f.name for f in a.low.glob('*.png'))
with torch.inference_mode():
 for name in names:
    x=rgb(a.low/name).astype(np.float32)/255; y=rgb(a.high/name).astype(np.float32)/255
    pred=model(torch.from_numpy(x.transpose(2,0,1).copy())[None]).clamp(0,1)[0].permute(1,2,0).numpy()
    rows.append(dict(image=name,psnr=float(u.PSNR(pred,y,False)),ssim=float(u.calculate_ssim(pred,y,border=0,gtmean=False))))
if not rows: raise ValueError('No images selected')
with a.output.open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
print(json.dumps({k:float(np.mean([r[k] for r in rows])) for k in ['psnr','ssim']},indent=2))
