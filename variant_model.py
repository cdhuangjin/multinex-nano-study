import sys
from pathlib import Path
from torch import nn
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from rep_model import ROOT,build_rep,build_student

class PartialRep(nn.Module):
    def __init__(self,full,arm):
        super().__init__()
        self.conv3x3=full.conv3x3; self.bn3x3=full.bn3x3; self.arm=arm
        if arm=='A': self.conv1x1=full.conv1x1; self.bn1x1=full.bn1x1
        elif arm=='B': self.bn_id=full.bn_id
        else: raise ValueError(arm)
    def forward(self,x):
        main=self.bn3x3(self.conv3x3(x))
        return main+(self.bn1x1(self.conv1x1(x)) if self.arm=='A' else self.bn_id(x))

def build(arm,seed):
    if arm=='baseline': return build_student(seed)
    model=build_rep(seed)
    if arm in ('A','B'):
        for branch in ('illum_mid_seq','chroma_mid_seq'):
            holder=getattr(model,branch)[0][1]
            holder.depthwise_conv=PartialRep(holder.depthwise_conv,arm)
    elif arm!='rep': raise ValueError(arm)
    return model
