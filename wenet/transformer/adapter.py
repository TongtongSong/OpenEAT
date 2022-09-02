# --------------------------------------------------------
# References:
# https://github.com/jxhe/unify-parameter-efficient-tuning
# --------------------------------------------------------

import math
import torch
import torch.nn as nn


class Adapter(nn.Module):
    def __init__(self,
                 d_model,
                 dropout_rate=0.1,
                 down_size = 64,
                 adapter_scalar=0.1):
        super().__init__()

        if adapter_scalar == -1:
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = adapter_scalar
        self.norm = nn.LayerNorm(d_model, eps=1e-12)
        self.down_proj = nn.Linear(d_model, down_size)
        self.relu = nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.up_proj = nn.Linear(down_size, d_model)

    def forward(self, x):
        x = self.norm(x)
        x = self.up_proj(self.dropout(self.relu(self.down_proj(x))))
        x = x * self.scale
        return x