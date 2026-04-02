import torch.nn as nn
import torch

__all__ = ['SharedMLP']


class Swish(nn.Module):
    def forward(self,x):
        return  x * torch.sigmoid(x)

class SharedMLP(nn.Module):
    def __init__(self, in_channels, out_channels, dim=1):
        super().__init__()
        self.dim = dim
        if dim == 1:
            conv = nn.Conv1d
            bn = nn.GroupNorm
        elif dim == 2:
            conv = nn.Conv2d
            bn = nn.GroupNorm
        else:
            raise ValueError
        if not isinstance(out_channels, (list, tuple)):
            out_channels = [out_channels]
        layers = []
        for oc in out_channels:
            layers.extend([
                conv(in_channels, oc, 1),
                bn(8, oc),
                Swish(),
            ])
            in_channels = oc
        self.layers = nn.Sequential(*layers)

    def _forward_tensor(self, inputs):
        if self.dim == 1 and inputs.dim() == 2:
            return self.layers(inputs.unsqueeze(-1)).squeeze(-1)
        return self.layers(inputs)

    def forward(self, inputs):
        if isinstance(inputs, (list, tuple)):
            return (self._forward_tensor(inputs[0]), *inputs[1:])
        else:
            return self._forward_tensor(inputs)
