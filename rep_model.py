"""Training-only parallel branches for Nano's two MSEF depthwise convolutions."""
import sys
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
from student import build_student

class RepBlock(nn.Module):
    def __init__(self, conv):
        super().__init__()
        c = conv.in_channels
        assert conv.out_channels == c and conv.groups == c and conv.kernel_size == (3, 3) and conv.bias is not None
        self.conv3x3 = conv
        self.bn3x3 = nn.BatchNorm2d(c)
        self.conv1x1 = nn.Conv2d(c, c, 1, groups=c, bias=True)
        self.bn1x1 = nn.BatchNorm2d(c)
        self.bn_id = nn.BatchNorm2d(c)

    def forward(self, x):
        return self.bn3x3(self.conv3x3(x)) + self.bn1x1(self.conv1x1(x)) + self.bn_id(x)

    @staticmethod
    def fold(conv, bn):
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        return conv.weight * scale[:, None, None, None], (conv.bias - bn.running_mean)*scale + bn.bias

    def deploy(self):
        if self.training:
            raise ValueError('BN folding requires evaluation mode')
        k3,b3 = self.fold(self.conv3x3, self.bn3x3)
        k1,b1 = self.fold(self.conv1x1, self.bn1x1)
        c = self.conv3x3.in_channels
        scale = self.bn_id.weight / torch.sqrt(self.bn_id.running_var + self.bn_id.eps)
        identity = torch.zeros_like(k3); identity[:, 0, 1, 1] = scale
        bias_id = self.bn_id.bias - self.bn_id.running_mean * scale
        result = nn.Conv2d(c,c,3,padding=1,groups=c,bias=True).to(k3.device, k3.dtype)
        with torch.no_grad():
            result.weight.copy_(k3 + F.pad(k1,(1,1,1,1)) + identity)
            result.bias.copy_(b3+b1+bias_id)
        return result

def build_rep(seed=110):
    model = build_student(seed)
    for branch in ('illum_mid_seq', 'chroma_mid_seq'):
        block = getattr(model, branch)[0][1]
        block.depthwise_conv = RepBlock(block.depthwise_conv)
    return model

def convert(model):
    model.eval()
    for branch in ('illum_mid_seq', 'chroma_mid_seq'):
        block = getattr(model, branch)[0][1]
        block.depthwise_conv = block.depthwise_conv.deploy()
    return model
