"""Check folding equivalence; optionally export trained models for CPU timing."""
import argparse, copy, json
import torch
from rep_model import build_rep, build_student, convert
p=argparse.ArgumentParser(); p.add_argument('--rep-checkpoint'); p.add_argument('--baseline-checkpoint'); p.add_argument('--export',action='store_true'); a=p.parse_args()
torch.set_num_threads(4); rep=build_rep(110).eval(); baseline=build_student(110).eval()
for model,path in [(rep,a.rep_checkpoint),(baseline,a.baseline_checkpoint)]:
 if path: model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['params'])
merged=convert(copy.deepcopy(rep)); torch.manual_seed(2026); x=torch.rand(1,3,256,256)
with torch.inference_mode():
 error=(rep(x)-merged(x)).abs().max().item()
assert sum(p.numel() for p in baseline.parameters())==696
assert sum(p.numel() for p in rep.parameters())==744
assert sum(p.numel() for p in merged.parameters())==696
assert error<1e-5,error
print(json.dumps(dict(baseline_parameters=696,training_parameters=744,merged_parameters=696,max_abs_error=error)))
if a.export:
 if not a.rep_checkpoint or not a.baseline_checkpoint: p.error('Export requires both trained checkpoints')
 for name,model in [('baseline',baseline),('merged',merged)]:
  torch.onnx.export(model,x,name+'.onnx',input_names=['input'],output_names=['output'],opset_version=17,dynamo=False)
 torch.save({'params':merged.state_dict()},'merged.pth')
