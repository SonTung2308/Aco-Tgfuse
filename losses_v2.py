"""
losses_v2.py — Loss cải tiến để gỡ trade-off SF↔SSIM↔SD.

Thay đổi chính so với code cũ:
  1. saliency_intensity_loss: thay target thô max(ir,vi) bằng tổ hợp có trọng số
     theo saliency cục bộ (VSM proxy). max(ir,vi) hay làm cháy sáng và đẩy xa SSIM.
  2. final_ssim_soft: thay torch.where nhị phân bằng trọng số mềm (sigmoid của hiệu
     std chuẩn hoá) → gradient mượt, hội tụ ổn định hơn.
  3. UncertaintyWeighter (Kendall et al.): tự cân các loss thay vì dò tay λ.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from loss import mssim, std   # tái dùng từ loss.py gốc


# ──────────────────────────────────────────────────────────────────────────
# Saliency weighting (VSM proxy — độ tương phản cục bộ)
# ──────────────────────────────────────────────────────────────────────────
def saliency_weights(ir, vi, ksize: int = 7, eps: float = 1e-6):
    """Trọng số per-pixel cho IR/VIS theo local std. Không cần gradient."""
    with torch.no_grad():
        def local_std(x):
            mu  = F.avg_pool2d(x, ksize, 1, ksize // 2)
            mu2 = F.avg_pool2d(x * x, ksize, 1, ksize // 2)
            return (mu2 - mu * mu).clamp_min(0).sqrt()
        s_ir, s_vi = local_std(ir), local_std(vi)
        w_ir = (s_ir + eps) / (s_ir + s_vi + 2 * eps)
        w_vi = 1.0 - w_ir
    return w_ir, w_vi


def saliency_intensity_loss(ir, vi, fused):
    """L1 tới target tổ hợp theo saliency — nhẹ hơn max(ir,vi), nâng MI/SD ổn hơn."""
    w_ir, w_vi = saliency_weights(ir, vi)
    target = w_ir * ir + w_vi * vi
    return F.l1_loss(fused, target)


# ──────────────────────────────────────────────────────────────────────────
# Soft-reference SSIM
# ──────────────────────────────────────────────────────────────────────────
def final_ssim_soft(ir, vi, fused, tau: float = 6.0, eps: float = 1e-6):
    s_ir = mssim(ir, fused)
    s_vi = mssim(vi, fused)
    std_ir, std_vi = std(ir), std(vi)
    # hiệu std chuẩn hoá ∈ [-1,1], sigmoid(tau··) cho trọng số mềm
    diff = (std_ir - std_vi) / (std_ir + std_vi + eps)
    w = torch.sigmoid(tau * diff)
    ssim = w * s_ir + (1.0 - w) * s_vi
    return ssim.mean()


def ssim_loss_soft(ir, vi, fused):
    return 1.0 - final_ssim_soft(ir, vi, fused)


# ──────────────────────────────────────────────────────────────────────────
# Gradient & edge
# ──────────────────────────────────────────────────────────────────────────
def _grad(x):
    kx = torch.tensor([[0, 0, 0], [-1, 0, 1], [0, 0, 0]],
                      dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    ky = torch.tensor([[0, -1, 0], [0, 0, 0], [0, 1, 0]],
                      dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    # eps=1.0 (không phải 1e-8): chặn gradient của sqrt khỏi nổ tại pixel gradient≈0,
    # đây là nguồn NaN trong backward khi output model có range rộng.
    return torch.sqrt(F.conv2d(x, kx, padding=1) ** 2 +
                      F.conv2d(x, ky, padding=1) ** 2 + 1.0)


def gradient_loss(ir, vi, fused):
    """
    CHỈ L1 đối xứng tới max-gradient. ĐÃ BỎ phạt bất đối xứng F.relu(g_max-g_fus):
    nó ép fused luôn 'sắc ít nhất bằng nguồn' kể cả ở vùng đáng lẽ mượt (trời, tán
    cây) → sinh đốm/ringing như thấy trong fused_06. L1 đối xứng đủ giữ edge mà
    không over-sharpen.
    """
    g_max = torch.max(_grad(ir), _grad(vi))
    return F.l1_loss(_grad(fused), g_max)


def edge_loss(ir, vi, fused):
    lap = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                       dtype=fused.dtype, device=fused.device).view(1, 1, 3, 3)
    e_ir  = F.conv2d(ir / 255.0,    lap, padding=1).abs()
    e_vi  = F.conv2d(vi / 255.0,    lap, padding=1).abs()
    e_fus = F.conv2d(fused / 255.0, lap, padding=1).abs()
    e_max = torch.max(e_ir, e_vi)
    return F.l1_loss(e_fus, e_max)   # bỏ phạt bất đối xứng (xem gradient_loss)


# ──────────────────────────────────────────────────────────────────────────
# Kendall uncertainty weighting — tự cân các loss
# ──────────────────────────────────────────────────────────────────────────
class UncertaintyWeighter(nn.Module):
    """
    L = Σ_i [ 0.5 · exp(-s_i) · L_i + 0.5 · s_i ],  s_i = log(σ_i^2) học được.
    Nhớ ĐƯA params của weighter vào optimizer.
    """
    def __init__(self, n_losses: int, lo: float = -2.0, hi: float = 2.0):
        super().__init__()
        self.log_var = nn.Parameter(torch.zeros(n_losses))
        self.lo, self.hi = lo, hi   # kẹp để không loss nào bị bóp về 0 / phình vô hạn

    def forward(self, losses):
        lv = self.log_var.clamp(self.lo, self.hi)
        total = 0.0
        for i, l in enumerate(losses):
            total = total + 0.5 * torch.exp(-lv[i]) * l + 0.5 * lv[i]
        return total

    @torch.no_grad()
    def weights(self):
        return [float(torch.exp(-0.5 * v)) for v in self.log_var]