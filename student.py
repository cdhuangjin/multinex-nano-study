from pathlib import Path
import torch, yaml
from official_adapter import load_file
ROOT=Path(__file__).resolve().parent

def build_student(seed):
    options=yaml.safe_load((ROOT/'code/Options/Multinex-Nano_LOL-v1.yaml').read_text(encoding='utf-8'))['network_g']
    options=dict(options)
    assert options.pop('type')=='MultinexNano'
    # Construction uses RNG internally; reset before every paired student.
    torch.manual_seed(seed)
    module=load_file('distillation_nano_upstream','basicsr/models/archs/MultinexNano_arch.py')
    model=module.MultinexNano(**options)
    assert sum(p.numel() for p in model.parameters())==696
    return model

