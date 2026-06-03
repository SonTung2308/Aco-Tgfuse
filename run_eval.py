"""
run_eval.py — Đánh giá model ACO-TGFuse: XUẤT ẢNH để nhìn mắt + METRIC chuẩn.

Cách dùng:
    python run_eval.py --model models_aco_v9/best_composite.model
    python run_eval.py --model models_aco_v9/best_ssim.model

Xuất:
    eval_out/<tên_model>/fused_XX.png        # ảnh fused — MỞ LÊN NHÌN trước tiên
    eval_out/<tên_model>/triptych_XX.png     # IR | VIS | Fused ghép ngang để so
    in ra bảng metric trung bình.

Metric tự chứa (không cần evaluate_final.py): SF, EN, SD, MI, SSIM-var, Qabf, AG.
Tất cả tính trên ảnh uint8 [0,255], khớp convention của TGFuse.
"""
import os, sys, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import torch
import cv2
from pathlib import Path

sys.path.insert(0, '/home/iec/vstung/TGFuse')
from net_aco_v5 import ACOFusionNet


# ── Metric (numpy, ảnh uint8 [0,255]) ──────────────────────────────────────
def m_SF(F):
    F = F.astype(np.float64)
    rf = np.diff(F, axis=1); cf = np.diff(F, axis=0)
    return float(np.sqrt((rf**2).mean() + (cf**2).mean()))

def m_SD(F):
    return float(F.astype(np.float64).std())

def m_EN(F):
    h = np.histogram(F, bins=256, range=(0, 256))[0].astype(np.float64)
    p = h / h.sum(); p = p[p > 0]
    return float(-(p * np.log2(p)).sum())

def m_AG(F):
    F = F.astype(np.float64)
    gx = np.diff(F, axis=1)[:-1, :]; gy = np.diff(F, axis=0)[:, :-1]
    return float(np.sqrt((gx**2 + gy**2) / 2).mean())

def _mi_pair(A, F):
    h = np.histogram2d(A.ravel(), F.ravel(), bins=256,
                       range=[[0, 256], [0, 256]])[0]
    pab = h / h.sum()
    pa = pab.sum(1, keepdims=True); pb = pab.sum(0, keepdims=True)
    nz = pab > 0
    return float((pab[nz] * np.log2(pab[nz] / (pa @ pb)[nz])).sum())

def m_MI(A, B, F):
    return _mi_pair(A, F) + _mi_pair(B, F)

def m_Qabf(A, B, F):
    # bản gọn của Xydeas-Petrovic
    def edge(x):
        x = x.astype(np.float64)
        gx = cv2.Sobel(x, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(x, cv2.CV_64F, 0, 1, ksize=3)
        g = np.sqrt(gx**2 + gy**2)
        a = np.arctan2(gy, gx + 1e-12)
        return g, a
    gA, aA = edge(A); gB, aB = edge(B); gF, aF = edge(F)
    def Q(gX, aX):
        Gm = np.minimum(gX, gF) / (np.maximum(gX, gF) + 1e-12)
        Am = 1 - np.abs(aX - aF) / (np.pi / 2)
        return (Gm * Am).clip(0, 1)
    QA = Q(gA, aA); QB = Q(gB, aB)
    wA = gA; wB = gB
    num = (QA * wA + QB * wB).sum()
    den = (wA + wB).sum() + 1e-12
    return float(num / den)

def m_SSIMvar(A, B, F):
    # tái dùng định nghĩa Var-SSIM của repo (chọn ref theo std cao hơn) — gọn
    from loss import mssim, std
    import torch as T
    def t(x): return T.from_numpy(x.astype(np.float32))[None, None].cuda()
    A_, B_, F_ = t(A), t(B), t(F)
    sA = mssim(A_, F_); sB = mssim(B_, F_)
    dA, dB = std(A_), std(B_)
    m1 = (dA - dB > 0).float()
    ss = m1 * sA + (1 - m1) * sB
    return float(ss.mean().item())


# ── Inference 1 cặp, trả ảnh fused uint8 đúng kích thước gốc ────────────────
@torch.no_grad()
def fuse(model, ir_path, vi_path):
    A = cv2.imread(str(ir_path), 0)   # IR
    B = cv2.imread(str(vi_path), 0)   # VIS
    H, W = A.shape
    pad = lambda x: cv2.copyMakeBorder(
        x, 0, (128 - H % 128) % 128, 0, (128 - W % 128) % 128, cv2.BORDER_REFLECT)
    Ap, Bp = pad(A), pad(B)
    it = torch.from_numpy(Ap.astype(np.float32))[None, None].cuda()
    vt = torch.from_numpy(Bp.astype(np.float32))[None, None].cuda()
    out = model(vt, it).cpu().squeeze().numpy()[:H, :W]
    return A, B, np.clip(out, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ir", default="/home/iec/vstung/TGFuse/images/Test_ir")
    ap.add_argument("--vi", default="/home/iec/vstung/TGFuse/images/Test_vi")
    ap.add_argument("--out", default="/home/iec/vstung/TGFuse/eval_out")
    ap.add_argument("--combo", default="0,3")
    args = ap.parse_args()

    tag = Path(args.model).stem
    outdir = Path(args.out) / tag
    outdir.mkdir(parents=True, exist_ok=True)

    model = ACOFusionNet().cuda()
    model.set_branches(tuple(int(x) for x in args.combo.split(",")))
    model.load_state_dict(torch.load(args.model, map_location='cpu'))
    model.eval()
    print(f"Loaded {args.model}  combo={args.combo}")

    ir_files = sorted(Path(args.ir).iterdir())
    vi_files = sorted(Path(args.vi).iterdir())
    rows = []
    for k, (irp, vip) in enumerate(zip(ir_files, vi_files), 1):
        A, B, Fu = fuse(model, irp, vip)
        cv2.imwrite(str(outdir / f"fused_{k:02d}.png"), Fu)
        trip = np.concatenate([A, B, Fu], axis=1)
        cv2.imwrite(str(outdir / f"triptych_{k:02d}.png"), trip)
        rows.append(dict(
            SF=m_SF(Fu), SD=m_SD(Fu), EN=m_EN(Fu), AG=m_AG(Fu),
            MI=m_MI(A, B, Fu), Qabf=m_Qabf(A, B, Fu), SSIMv=m_SSIMvar(A, B, Fu)))
        print(f"  [{k:02d}] SF={rows[-1]['SF']:.3f} SD={rows[-1]['SD']:.2f} "
              f"EN={rows[-1]['EN']:.3f} MI={rows[-1]['MI']:.3f} "
              f"Qabf={rows[-1]['Qabf']:.4f} SSIMv={rows[-1]['SSIMv']:.4f}")

    print("\n" + "=" * 56)
    print(f"MEAN over {len(rows)} pairs  ({tag})")
    print("=" * 56)
    for key in ["SF", "SD", "EN", "AG", "MI", "Qabf", "SSIMv"]:
        vals = [r[key] for r in rows]
        print(f"  {key:6s} = {np.mean(vals):8.4f}  (±{np.std(vals):.3f})")
    print(f"\nẢnh xuất tại: {outdir}/  → MỞ triptych_*.png NHÌN TRƯỚC.")
    print("Nếu fused có viền giả / nhiễu hạt / 'rỗ' → SF cao là giả, chỉnh loss.")


if __name__ == "__main__":
    main()
