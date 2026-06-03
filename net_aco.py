import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple
from t2t_vit import Channel, Spatial


class ConvLayer(nn.Module):
    def __init__(self, ic, oc, k=1, s=1, last=False):
        super().__init__()
        self.pad  = nn.ReflectionPad2d(k // 2)
        self.conv = nn.Conv2d(ic, oc, k, s)
        self.last = last
    def forward(self, x):
        out = self.conv(self.pad(x.cuda()))
        return out if self.last else F.leaky_relu(out, inplace=True)


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1  = ConvLayer(c, c, 3)
        self.bn1 = nn.BatchNorm2d(c)
        self.c2  = ConvLayer(c, c, 3)
        self.bn2 = nn.BatchNorm2d(c)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.bn2(self.c2(self.relu(self.bn1(self.c1(x))))) + x)


class Encoder(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.net = nn.Sequential(ConvLayer(ic, oc), ResBlock(oc), ConvLayer(oc, oc))
    def forward(self, x):
        return self.net(x)


class ACOFusionNet(nn.Module):
    def __init__(self, channels: int = 64):
        super().__init__()
        C = channels
        self.down1 = nn.AvgPool2d(2)
        self.down2 = nn.AvgPool2d(4)
        self.down3 = nn.AvgPool2d(8)
        self.conv_in  = ConvLayer(2, 2)
        self.en0      = Encoder(2, C)
        self.en1      = Encoder(C, C)
        self.en2      = Encoder(C, C)
        self.en3      = Encoder(C, C)
        self.proj     = ConvLayer(2 * C, C, k=1)
        self.ctrans   = Channel(size=32, embed_dim=128, patch_size=16, channel=C)
        self.strans   = Spatial(size=256, embed_dim=2048, patch_size=4, channel=C)
        self.conv_out = ConvLayer(C, 1, last=True)
        self._selected: Tuple[int, int] = (0, 3)

    def set_branches(self, selected: Tuple[int, int]):
        self._selected = tuple(sorted(selected))

    def forward(self, vi: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        x  = self.conv_in(torch.cat([vi, ir], dim=1))
        x0 = self.en0(x)
        x1 = self.en1(self.down1(x0))
        x2 = self.en2(self.down1(x1))
        x3 = self.en3(self.down1(x2))
        feats = [x0, x1, x2, x3]

        b0, b1 = self._selected
        f0s = F.adaptive_avg_pool2d(feats[b0], (32, 32))
        f1s = F.adaptive_avg_pool2d(feats[b1], (32, 32))
        merged    = self.proj(torch.cat([f0s, f1s], dim=1))
        mask_small = self.strans(self.ctrans(merged))   # [B, C, 32, 32]

        # match mask to each feature size (supports any input resolution)
        def match(feat):
            return F.interpolate(mask_small, size=feat.shape[2:],
                                 mode='bilinear', align_corners=True)

        x0r = feats[0] * match(feats[0])
        x1r = feats[1] * match(feats[1])
        x2r = feats[2] * match(feats[2])
        x3r = feats[3] * match(feats[3])

        # upsample all to original size then sum
        H, W = feats[0].shape[2:]
        def up(f):
            return F.interpolate(f, size=(H, W), mode='bilinear', align_corners=True)

        out = up(x3r) + up(x2r) + up(x1r) + x0r
        return self.conv_out(out)
