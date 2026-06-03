"""
net_aco_v5.py — Mạng "production" cho cấu hình tốt nhất (mặc định en0+en3).

Tích hợp đầy đủ các cải tiến đã bàn:
  * CrossBranchAttention (bản sạch) giữa 2 nhánh ACO chọn
  * Relationship map 1-kênh từ Channel→Spatial transformer (giữ ý đồ paper:
    nén kênh về 1 để được "spatial relationship map")
  * SpatialGate per-pixel thay BranchGate global
  * AdaptiveScaleWeights: trộn 4 scale theo nội dung ảnh
  * EdgeResidual ở output

Tương thích interface cũ: set_branches((0,3)), forward(vi, ir).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from t2t_vit import Channel, Spatial
from aco_modules import (ConvLayer, Encoder, SpatialGate, CrossBranchAttention,
                         AdaptiveScaleWeights, EdgeResidual)


class ACOFusionNet(nn.Module):
    def __init__(self, channels: int = 64, aco_boost: float = 0.5,
                 use_cross: bool = True, use_edge: bool = True,
                 use_wsum: bool = True):
        super().__init__()
        C = channels
        self.aco_boost = aco_boost
        self.use_cross = use_cross
        self.use_edge = use_edge
        self.use_wsum = use_wsum

        self.down = nn.AvgPool2d(2)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.up2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.up3 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)

        self.conv_in = ConvLayer(2, 2)
        self.en0 = Encoder(2, C)
        self.en1 = Encoder(C, C)
        self.en2 = Encoder(C, C)
        self.en3 = Encoder(C, C)

        # Transformer giữ nguyên cấu hình baseline (channel rồi spatial)
        self.ctrans3 = Channel(size=32, embed_dim=128,  patch_size=16, channel=C)
        self.strans3 = Spatial(size=256, embed_dim=1024 * 2, patch_size=4, channel=C)

        self.cross = CrossBranchAttention(C, heads=4)

        self.sgate = nn.ModuleList([SpatialGate(C) for _ in range(4)])
        self.scale_weights = AdaptiveScaleWeights(C)
        self.edge = EdgeResidual()
        self.conv_out = ConvLayer(C, 1, last=True)

        self._selected: Tuple[int, int] = (0, 3)

    def set_branches(self, selected: Tuple[int, int]):
        self._selected = tuple(sorted(selected))

    def forward(self, vi: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([vi, ir], dim=1)
        x = self.conv_in(inp)
        x0 = self.en0(x)
        x1 = self.en1(self.down(x0))
        x2 = self.en2(self.down(x1))
        x3 = self.en3(self.down(x2))
        feats = [x0, x1, x2, x3]

        # Cross-attention giữa 2 nhánh được chọn
        if self.use_cross:
            a, b = self._selected
            fa, fb = self.cross(feats[a], feats[b])
            feats[a], feats[b] = fa, fb
            x3 = feats[3]

        # Relationship map 1-kênh
        m = self.strans3(self.ctrans3(x3))          # [B, 1, 32, 32]

        x3m = m
        x2m = self.up1(x3m)
        x1m = self.up1(x2m) + self.up2(x3m)
        x0m = self.up1(x1m) + self.up2(x2m) + self.up3(x3m)
        masks = [x0m, x1m, x2m, x3m]

        r = [feats[k] * masks[k] for k in range(4)]
        r = [self.sgate[k](r[k]) for k in range(4)]

        # ACO prior boost cho 2 nhánh được chọn
        w = [1.0, 1.0, 1.0, 1.0]
        for k in self._selected:
            w[k] += self.aco_boost
        r = [r[k] * w[k] for k in range(4)]

        H, W = feats[0].shape[2:]
        up = [r[0],
              F.interpolate(r[1], size=(H, W), mode='bilinear', align_corners=False),
              F.interpolate(r[2], size=(H, W), mode='bilinear', align_corners=False),
              F.interpolate(r[3], size=(H, W), mode='bilinear', align_corners=False)]

        if self.use_wsum:
            sw = self.scale_weights(up)             # [B, 4]
            B = up[0].shape[0]
            agg = sum(up[k] * sw[:, k].view(B, 1, 1, 1) for k in range(4))
        else:
            agg = up[0] + up[1] + up[2] + up[3]

        out = self.conv_out(agg)
        if self.use_edge:
            out = out + self.edge(inp)
        return out
