"""Load unchanged upstream modules without importing the unused training stack."""
import importlib.util
from pathlib import Path
import sys
import types

import torch
import yaml

ROOT = Path(__file__).resolve().parent
CODE = ROOT / 'code'


def load_file(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, CODE / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def official_metrics():
    # Package shells suppress only BasicSR's eager training-module discovery.
    # All numerical functions below execute the original files, without edits.
    for name, directory in [('basicsr', 'basicsr'),
                            ('basicsr.metrics', 'basicsr/metrics'),
                            ('basicsr.utils', 'basicsr/utils')]:
        package = types.ModuleType(name)
        package.__path__ = [str(CODE / directory)]
        sys.modules[name] = package
    load_file('basicsr.utils.matlab_functions', 'basicsr/utils/matlab_functions.py')
    load_file('basicsr.metrics.metric_util', 'basicsr/metrics/metric_util.py')
    load_file('basicsr.metrics.psnr_ssim', 'basicsr/metrics/psnr_ssim.py')
    return load_file('multinex_official_image_utils', 'Enhancement/utils.py')


def load_model(item, device):
    module = load_file(item['class_name'] + '_official', item['architecture'])
    options = yaml.safe_load((CODE / item['options']).read_text(encoding='utf-8'))
    network = dict(options['network_g'])
    assert network.pop('type') == item['class_name']
    model = getattr(module, item['class_name'])(**network)
    checkpoint = torch.load(CODE / item['weights'], map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['params'], strict=True)
    return model.to(device).eval()
