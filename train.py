"""Paired, fixed-endpoint seed confirmation of the existing N0 and Rep-Nano."""
import argparse, hashlib, json, random, sys, time
from pathlib import Path
import torch, yaml

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from variant_model import ROOT, build
sys.path.insert(0,str(ROOT/'experiment_design'))
from train_controlled import batch, criterion, validate
from official_adapter import load_file

def main():
    p=argparse.ArgumentParser(); p.add_argument('--seed',type=int,choices=[101,102,103,104,110],required=True); p.add_argument('--arm',choices=['baseline','rep','A','B'],required=True); args=p.parse_args()
    config=ROOT/'config.yaml'
    cfg=yaml.safe_load(config.read_text(encoding='utf-8')); cfg['manual_seed']=args.seed
    assert cfg['train']['total_iter']==3000 and cfg['datasets']['train']['batch_size_per_gpu']==2
    out=HERE/'runs'/f'{args.arm}-s{args.seed}'; out.mkdir(parents=True,exist_ok=False)
    try:
        torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
        random.seed(args.seed); model=build(args.arm,args.seed).cuda().train()
        loss_fn,vgg_sha=criterion(cfg)
        optimizer=torch.optim.Adam(model.parameters(),**{k:v for k,v in cfg['train']['optim_g'].items() if k!='type'})
        mod=load_file('replication_scheduler','basicsr/models/lr_scheduler.py'); scheduler=mod.CosineAnnealingRestartCyclicLR(optimizer,**{k:v for k,v in cfg['train']['scheduler'].items() if k!='type'})
        split=json.loads((ROOT/'split_local.json').read_text(encoding='utf-8'))
        provenance={'arm':args.arm,'seed':args.seed,'base_config':str(config),'base_config_sha256':hashlib.sha256(config.read_bytes()).hexdigest(),'vgg_sha256':vgg_sha,'train_parameters':sum(p.numel() for p in model.parameters())}
        (out/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
        best={'psnr':-1,'step':0}; started=time.monotonic()
        for step in range(1,3001):
            if step>1: scheduler.step()
            x,y,sig=batch(cfg,split,step); optimizer.zero_grad(set_to_none=True)
            pred=model(x); loss=loss_fn(pred,y)
            if not torch.isfinite(loss): raise FloatingPointError(f'nonfinite loss at {step}')
            loss.backward(); norm=torch.nn.utils.clip_grad_norm_(model.parameters(),.01,error_if_nonfinite=True); optimizer.step()
            with (out/'train.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps({'step':step,'loss':loss.item(),'gradient_norm':float(norm),'lr':optimizer.param_groups[0]['lr'],'batch_signature':sig})+'\n')
            if step%500==0:
                metrics=validate(model,split,out,step)
                if metrics['psnr']>best['psnr']: best={'step':step,**metrics}
                state={'step':step,'params':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'best':best}
                torch.save(state,out/'latest.pth')
                if best['step']==step: torch.save(state,out/'best.pth')
                with (out/'validation.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps({'step':step,**metrics})+'\n')
                print(json.dumps({'step':step,**metrics}),flush=True)
        (out/'summary.json').write_text(json.dumps({'status':'COMPLETE','arm':args.arm,'seed':args.seed,'step':3000,'final':metrics,'best':best,'elapsed_seconds':time.monotonic()-started},indent=2),encoding='utf-8')
    except Exception as e:
        (out/'FAILED.json').write_text(json.dumps({'type':type(e).__name__,'error':str(e)},indent=2),encoding='utf-8'); raise

if __name__=='__main__': main()
