"""
normalize
"""

import math
import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        scale = torch.rsqrt(x.pow(2).mean(dim = -1, keepdim = True) + self.eps)
        return self.weight * scale * x

