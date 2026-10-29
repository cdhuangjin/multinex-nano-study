"""Bind the published split to an existing LOL-v1 copy; never download images."""
import argparse, hashlib, json
from pathlib import Path
import yaml
p=argparse.ArgumentParser()
p.add_argument('--data-root',type=Path,required=True)
p.add_argument('--torch-home',type=Path,required=True)
a=p.parse_args(); root=Path(__file__).resolve().parent
s=json.loads((root/'split_manifest.json').read_text())
for pair in s['files'].values():
    for item in pair.values():
        path=a.data_root/item['path']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:
            raise ValueError(f'Dataset checksum mismatch: {path}')
        item['path']=str(path.resolve())
assert len(s['train'])==435 and len(s['dev'])==50 and not set(s['train'])&set(s['dev'])
(root/'split_local.json').write_text(json.dumps(s,indent=2))
c=yaml.safe_load((root/'config.yaml').read_text()); c['experiment']['torch_home']=str(a.torch_home.resolve())
(root/'config.yaml').write_text(yaml.safe_dump(c,sort_keys=False))
print('Verified 485 image pairs; local paths configured.')
