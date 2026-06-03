"""
aco_modules.py — Building blocks dùng chung cho ACO-TGFuse (bản cải tiến).

Khác với code cũ:
  * ConvLayer KHÔNG ép .cuda() bên trong forward nữa (để caller quản lý device,
    tránh bug CPU/GPU và cho phép test). Nhớ gọi model.cuda() + input.cuda().
  * CrossBranchAttention được viết lại sạch (residual + LayerNorm đúng chuẩn),
    thay cho bản rối trong net_aco_v4.py.
  * SpatialGate: gate THEO TỪNG PIXEL thay cho BranchGate global (GAP) cũ.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


# ──────────────────────────────────────────────────────────────────────────
# Conv / Res / Encoder (giữ tương thích với net.py gốc)
# ──────────────────────────────────────────────────────────────────────────
class ConvLayer(nn.Module):
    def __init__(self, ic, oc, k=1, s=1, last=False):
        super().__init__()
        self.pad  = nn.ReflectionPad2d(k // 2)
        self.conv = nn.Conv2d(ic, oc, k, s)
        self.last = last

    def forward(self, x):
        out = self.conv(self.pad(x))
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


# ──────────────────────────────────────────────────────────────────────────
# Gating
# ──────────────────────────────────────────────────────────────────────────
class SpatialGate(nn.Module):
    """
    Gate THEO TỪNG PIXEL: sinh mask 1 kênh ∈ (0,1) từ feature rồi nhân vào.
    Khác BranchGate cũ (dùng GAP → 1 hệ số cho cả ảnh), gate này giữ được
    chọn lọc theo vị trí: vùng người nóng (IR) vs vùng texture (VIS) được
    đối xử khác nhau — khớp đúng với cái Var-SSIM loss đang muốn.
    """
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.body = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.body(x)


class ChannelGate(nn.Module):
    """Gate kênh global (giữ lại để so sánh trong supernet)."""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 4)), nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 4), channels), nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.fc(self.gap(x).view(b, c)).view(b, c, 1, 1)
        return x * w


# ──────────────────────────────────────────────────────────────────────────
# CrossBranchAttention — VIẾT LẠI SẠCH
# ──────────────────────────────────────────────────────────────────────────
class CrossBranchAttention(nn.Module):
    """
    Cross-attention HAI CHIỀU giữa 2 feature map ở (có thể) độ phân giải khác nhau.
    Cả hai được đưa về kích thước nhỏ hơn để tính attention; mỗi nhánh nhận lại
    residual update đúng độ phân giải gốc của nó.

    Thay thế bản trong net_aco_v4.py (chuỗi flatten/transpose/reshape_as lồng nhau
    quanh LayerNorm, rất dễ sai hình học tensor → đó mới là lý do v7 'thất bại',
    không phải do ý tưởng cross-attention sai).
    """
    def __init__(self, dim, heads=4, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % heads == 0, "dim phải chia hết cho heads"
        self.heads = heads
        self.scale = (dim // heads) ** -0.5

        self.norm_a = nn.LayerNorm(dim)
        self.norm_b = nn.LayerNorm(dim)
        self.q_a = nn.Linear(dim, dim);  self.kv_b = nn.Linear(dim, dim * 2)
        self.q_b = nn.Linear(dim, dim);  self.kv_a = nn.Linear(dim, dim * 2)
        self.proj_a = nn.Linear(dim, dim)
        self.proj_b = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    @staticmethod
    def _tokens(x):
        b, c, h, w = x.shape
        return x.flatten(2).transpose(1, 2), (b, c, h, w)   # [B, N, C]

    @staticmethod
    def _map(t, shape):
        b, c, h, w = shape
        return t.transpose(1, 2).reshape(b, c, h, w)

    def _mha(self, q, k, v):
        b, nq, c = q.shape
        nk = k.shape[1]
        h, d = self.heads, c // self.heads
        q = q.reshape(b, nq, h, d).permute(0, 2, 1, 3)
        k = k.reshape(b, nk, h, d).permute(0, 2, 1, 3)
        v = v.reshape(b, nk, h, d).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(-1))
        out = (attn @ v).permute(0, 2, 1, 3).reshape(b, nq, c)
        return out

    def forward(self, fa, fb):
        # đưa cả hai về kích thước nhỏ hơn để tính attention
        size = (min(fa.shape[2], fb.shape[2]), min(fa.shape[3], fb.shape[3]))
        fa_s = F.adaptive_avg_pool2d(fa, size)
        fb_s = F.adaptive_avg_pool2d(fb, size)

        ta, sa = self._tokens(fa_s)
        tb, sb = self._tokens(fb_s)
        ta_n, tb_n = self.norm_a(ta), self.norm_b(tb)

        # a <- b
        kb, vb = self.kv_b(tb_n).chunk(2, dim=-1)
        ta = ta + self.proj_drop(self.proj_a(self._mha(self.q_a(ta_n), kb, vb)))
        # b <- a
        ka, va = self.kv_a(ta_n).chunk(2, dim=-1)
        tb = tb + self.proj_drop(self.proj_b(self._mha(self.q_b(tb_n), ka, va)))

        out_a_s = self._map(ta, sa)
        out_b_s = self._map(tb, sb)

        # residual về đúng độ phân giải gốc
        out_a = fa + F.interpolate(out_a_s - fa_s, size=fa.shape[2:],
                                   mode='bilinear', align_corners=False)
        out_b = fb + F.interpolate(out_b_s - fb_s, size=fb.shape[2:],
                                   mode='bilinear', align_corners=False)
        return out_a, out_b


# ──────────────────────────────────────────────────────────────────────────
# Content-adaptive scale weighting & edge residual
# ──────────────────────────────────────────────────────────────────────────
class AdaptiveScaleWeights(nn.Module):
    """Học trọng số 4 nhánh theo nội dung ảnh (thay cho cộng đều)."""
    def __init__(self, channels):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels * 4, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 4), nn.Softmax(dim=1),
        )

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        pooled = torch.cat([self.gap(f).flatten(1) for f in feats], dim=1)
        return self.fc(pooled)   # [B, 4]


class EdgeResidual(nn.Module):
    """Trích edge từ ảnh input (2 kênh) → cộng nhẹ vào output, giúp Qabf."""
    def __init__(self):
        super().__init__()
        self.extract = nn.Sequential(
            ConvLayer(2, 8, 3), ConvLayer(8, 1, 3, last=True)
        )
        self.weight = nn.Parameter(torch.tensor(0.1))

    def forward(self, inp):
        return self.weight * self.extract(inp)
