"""
ffn
"""

import math
import torch
from torch import nn
from torch.nn import functional as F

class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.up = nn.Linear(dim, 2 * hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        a, b = self.up(x).chunk(2, -1)
        return self.down(F.silu(a) * b)
