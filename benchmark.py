import json, sys, time, platform, statistics
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'vendor'))
import onnxruntime as ort
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

opts=ort.SessionOptions(); opts.intra_op_num_threads=1; opts.inter_op_num_threads=1
sessions={k:ort.InferenceSession(str(HERE/f'{k}.onnx'),opts,providers=['CPUExecutionProvider']) for k in ('baseline','merged')}
x=np.random.default_rng(2026).random((1,3,256,256),dtype=np.float32)
for s in sessions.values():
    for _ in range(20): s.run(None,{'input':x})
rounds=[]
for i in range(10):
    row={'round':i+1,'order': ['baseline','merged'] if i%2==0 else ['merged','baseline']}
    for name in row['order']:
        values=[]
        for _ in range(100):
            t=time.perf_counter_ns(); sessions[name].run(None,{'input':x}); values.append((time.perf_counter_ns()-t)/1e6)
        row[name]={'mean_ms':statistics.mean(values),'samples_ms':values}
    row['paired_delta_ms']=row['merged']['mean_ms']-row['baseline']['mean_ms']; rounds.append(row)
names=['baseline','merged']; avgs=[statistics.mean(r[n]['mean_ms'] for r in rounds) for n in names]; sd=[statistics.stdev(r[n]['mean_ms'] for r in rounds) for n in names]
result={'input':[1,3,256,256],'threads':1,'warmup_per_model':20,'rounds':rounds,'rounds_count':10,'iterations_per_block':100,'baseline_mean_ms':avgs[0],'baseline_sd_round_means_ms':sd[0],'merged_mean_ms':avgs[1],'merged_sd_round_means_ms':sd[1],'paired_delta_mean_ms':statistics.mean(r['paired_delta_ms'] for r in rounds),'paired_delta_sd_ms':statistics.stdev(r['paired_delta_ms'] for r in rounds),'relative_delta_percent':100*(avgs[1]/avgs[0]-1),'runtime':ort.__version__,'platform':platform.platform(),'processor':platform.processor(),'note':'rounds share a machine/session; variability does not establish population-level significance; same inference architecture'}
(HERE/'benchmark_alternating.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
fig,ax=plt.subplots(figsize=(5.8,4)); ax.bar(['Baseline','Rep-Nano'],avgs,yerr=sd,capsize=5,color=['#527695','#d27b46']); ax.set_ylabel('CPU latency (ms)'); ax.set_title('ONNX Runtime, 256 脳 256, 1 thread'); ax.text(.5,-.18,'Mean 卤 sample SD of 10 round means',ha='center',transform=ax.transAxes,fontsize=9); fig.tight_layout(); fig.savefig(HERE/'cpu_latency.png',dpi=220); fig.savefig(HERE/'cpu_latency.pdf'); plt.close(fig)
print(json.dumps({k:v for k,v in result.items() if k!='rounds'},indent=2))
