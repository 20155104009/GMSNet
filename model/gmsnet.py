
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce


class MSConv(nn.Module):
    def __init__(self, planes, ratio=0.25):
        super(MSConv, self).__init__()
        out_planes = int(planes * ratio)
        self.conv1 = nn.Conv2d(planes, planes-out_planes, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(planes, out_planes, kernel_size=3, padding=2, dilation=2)

    def forward(self, x):
        return torch.cat([self.conv1(x), self.conv2(x)], dim=1)


class ResLayer(nn.Module):
    def __init__(self, planes):
        super(ResLayer, self).__init__()
        self.conv1 = MSConv(planes)
        self.prelu = nn.PReLU()
        self.conv2 = MSConv(planes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.prelu(out)
        out = self.conv2(out)

        out = out + x
        return out


class ResGroup(nn.Module):
    def __init__(self, planes, num_groups):
        super(ResGroup, self).__init__()

        self.num_blocks = num_groups[0]

        self.blocks = nn.ModuleList()
        for _ in range(self.num_blocks):
            if len(num_groups) > 1:
                self.blocks.append(ResGroup(planes, num_groups[1:]))
            else:
                self.blocks.append(ResLayer(planes))

        self.fuse = nn.Conv2d(self.num_blocks*planes, planes, kernel_size=1)

    def forward(self, x):
        residual = x

        out_list = []
        for i in range(self.num_blocks):
            x = self.blocks[i](x)
            out_list.append(x)

        out = self.fuse(torch.cat(out_list, dim=1))

        out = out + residual
        return out


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.planes = 64
        self.num_groups = [4, 4, 4]

        self.inc = nn.Sequential(
            nn.Conv2d(3, self.planes, kernel_size=3, padding=1),
            nn.PReLU()
        )

        self.blocks = ResGroup(self.planes, self.num_groups)

        self.outc = nn.Conv2d(self.planes, 3, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=np.sqrt(2 / (m.weight.shape[0] * np.prod(m.weight.shape[2:]))) * reduce(lambda x, y: x * y, self.num_groups) ** (-0.5))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        residual = x
        x = self.inc(x)
        identity = x

        out = self.blocks(x)

        out = out - identity
        out = self.outc(out)
        out = out + residual

        return out
        