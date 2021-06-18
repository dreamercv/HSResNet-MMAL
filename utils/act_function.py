import torch
from torch import nn
from torch.nn import functional as f

class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()
        print("load activate function: Mish")

    def forward(self, x):
        return x * torch.tanh(f.softplus(x))