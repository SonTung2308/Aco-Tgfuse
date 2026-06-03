"""
eval_datasets.py — Đánh giá model trên NHIỀU dataset, xuất bảng + LaTeX cho bài báo.

Chạy model qua từng dataset (cấu trúc ir/ + vi/), tính bộ metric đầy đủ chuẩn IVIF,
in bảng mean ± std cho từng dataset, và xuất file .tex sẵn để dán vào bài.

Bộ metric (chuẩn dùng trong bảng IVIF tạp chí):
  EN, SF, SD, MI, SCD, VIF, Qabf, AG, MS-SSIM, SSIMv
Tất cả "càng cao càng tốt".

Dùng:
  python eval_datasets.py \
    --model models_aco_v9/best_composite.model \
    --datasets datasets_eval/TNO datasets_eval/RoadScene datasets_eval/MSRS \
    --out paper_results
"""
import os, sys, argparse, json
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import torch
import cv2
from pathlib import Path

sys.path.insert(0, '/home/iec/vstung/TGFuse')
from net_aco_v5 import ACOFusionNet


IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
NORM = False   # bật bằng --norm: min-max output về [0,255] (công bằng với CDDFuse)


# ── Metric đầy đủ (numpy, ảnh uint8 [0,255]) ───────────────────────────────
def m_EN(F):
    h = np.histogram(F, 256, (0, 256))[0].astype(np.float64)
    p = h / h.sum(); p = p[p > 0]
    return float(-(p * np.log2(p)).sum())

def m_SF(F):
    F = F.astype(np.float64)
    return float(np.sqrt((np.diff(F, axis=1)**2).mean() + (np.diff(F, axis=0)**2).mean()))

def m_SD(F):
    return float(F.astype(np.float64).std())

def m_AG(F):
    F = F.astype(np.float64)
    gx = np.diff(F, axis=1)[:-1, :]; gy = np.diff(F, axis=0)[:, :-1]
    return float(np.sqrt((gx**2 + gy**2) / 2).mean())

def _mi(A, F):
    h = np.histogram2d(A.ravel(), F.ravel(), 256, [[0, 256], [0, 256]])[0]
    p = h / h.sum(); pa = p.sum(1, keepdims=True); pb = p.sum(0, keepdims=True)
    nz = p > 0
    return float((p[nz] * np.log2(p[nz] / (pa @ pb)[nz])).sum())

def m_MI(A, B, F):
    return _mi(A, F) + _mi(B, F)

def m_SCD(A, B, F):
    """
    Sum of Correlations of Differences (Aslantas & Bendes 2015).
    Ý tưởng: phần fused 'thêm' so với nguồn này phải tương quan với nguồn kia.
      D_A = F - B  (phần đóng góp của A vào F)  → tương quan với A
      D_B = F - A  (phần đóng góp của B vào F)  → tương quan với B
    SCD = r(D_A, A) + r(D_B, B). Fusion tốt thường ~1.3-1.8.
    """
    A = A.astype(np.float64); B = B.astype(np.float64); F = F.astype(np.float64)
    def r(x, y):
        x = x - x.mean(); y = y - y.mean()
        d = np.sqrt((x*x).sum() * (y*y).sum())
        return float((x*y).sum() / (d + 1e-12))
    return r(F - B, A) + r(F - A, B)

def m_VIF(A, B, F):
    """
    VIFF (Han et al. 2013) — Visual Information Fidelity for Fusion.
    Bản này tính trên TỪNG block với GSM model, đã kiểm chứng cho dải hợp lý (~0.3-1+).
    """
    def vif_pair(ref, dist, sigma_nsq=2.0):
        ref = ref.astype(np.float64); dist = dist.astype(np.float64)
        eps = 1e-10; num = 0.0; den = 0.0
        for scale in range(1, 5):
            N = 2 ** (4 - scale + 1) + 1
            win = cv2.getGaussianKernel(N, N / 5.0)
            win = win @ win.T
            if scale > 1:
                ref  = cv2.filter2D(ref,  -1, win, borderType=cv2.BORDER_REFLECT)[::2, ::2]
                dist = cv2.filter2D(dist, -1, win, borderType=cv2.BORDER_REFLECT)[::2, ::2]
            mu1 = cv2.filter2D(ref,  -1, win, borderType=cv2.BORDER_REFLECT)
            mu2 = cv2.filter2D(dist, -1, win, borderType=cv2.BORDER_REFLECT)
            mu1_sq, mu2_sq, mu1_mu2 = mu1*mu1, mu2*mu2, mu1*mu2
            s1  = cv2.filter2D(ref*ref,  -1, win, borderType=cv2.BORDER_REFLECT) - mu1_sq
            s2  = cv2.filter2D(dist*dist,-1, win, borderType=cv2.BORDER_REFLECT) - mu2_sq
            s12 = cv2.filter2D(ref*dist, -1, win, borderType=cv2.BORDER_REFLECT) - mu1_mu2
            s1 = np.maximum(s1, 0); s2 = np.maximum(s2, 0)
            g = s12 / (s1 + eps)
            sv_sq = s2 - g * s12
            g = np.where(s1 < eps, 0.0, g)
            sv_sq = np.where(s1 < eps, s2, sv_sq)
            s1 = np.where(s1 < eps, 0.0, s1)
            sv_sq = np.maximum(sv_sq, eps)
            num += np.sum(np.log10(1.0 + (g*g) * s1 / (sv_sq + sigma_nsq)))
            den += np.sum(np.log10(1.0 + s1 / sigma_nsq))
        return num / (den + eps)
    # VIFF: trung bình có trọng số 2 nguồn (xấp xỉ — đủ để xếp hạng)
    return float(0.5 * (vif_pair(A, F) + vif_pair(B, F)))

def _ssim_map(x, y, win, C1, C2):
    mu1 = cv2.filter2D(x, -1, win); mu2 = cv2.filter2D(y, -1, win)
    m1, m2, m12 = mu1*mu1, mu2*mu2, mu1*mu2
    s1 = cv2.filter2D(x*x, -1, win) - m1
    s2 = cv2.filter2D(y*y, -1, win) - m2
    s12 = cv2.filter2D(x*y, -1, win) - m12
    return ((2*m12 + C1)*(2*s12 + C2)) / ((m1 + m2 + C1)*(s1 + s2 + C2) + 1e-12)

def m_MSSSIM(A, B, F):
    # MS-SSIM trung bình của (A,F) và (B,F), 5 thang
    def msssim(ref, dist):
        ref = ref.astype(np.float64); dist = dist.astype(np.float64)
        k = cv2.getGaussianKernel(11, 1.5); win = k @ k.T
        C1 = (0.01*255)**2; C2 = (0.03*255)**2
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        mssim = []
        for i in range(5):
            sm = _ssim_map(ref, dist, win, C1, C2)
            mssim.append(np.clip(sm.mean(), 0, 1))
            ref = cv2.resize(ref, (max(ref.shape[1]//2,1), max(ref.shape[0]//2,1)))
            dist = cv2.resize(dist, (max(dist.shape[1]//2,1), max(dist.shape[0]//2,1)))
        return float(np.prod([m**w for m, w in zip(mssim, weights)]))
    return 0.5 * (msssim(A, F) + msssim(B, F))

def m_Qabf(A, B, F):
    def edge(x):
        x = x.astype(np.float64)
        gx = cv2.Sobel(x, cv2.CV_64F, 1, 0, 3); gy = cv2.Sobel(x, cv2.CV_64F, 0, 1, 3)
        return np.sqrt(gx**2 + gy**2), np.arctan2(gy, gx + 1e-12)
    gA, aA = edge(A); gB, aB = edge(B); gF, aF = edge(F)
    def Q(gX, aX):
        Gm = np.minimum(gX, gF) / (np.maximum(gX, gF) + 1e-12)
        Am = 1 - np.abs(aX - aF) / (np.pi/2)
        return (Gm * Am).clip(0, 1)
    QA, QB = Q(gA, aA), Q(gB, aB)
    return float((QA*gA + QB*gB).sum() / (gA + gB).sum() + 1e-12)

def m_SSIMv(A, B, F):
    from loss import mssim, std
    def t(x): return torch.from_numpy(x.astype(np.float32))[None, None].cuda()
    A_, B_, F_ = t(A), t(B), t(F)
    sA, sB = mssim(A_, F_), mssim(B_, F_)
    dA, dB = std(A_), std(B_)
    m1 = (dA - dB > 0).float()
    return float((m1*sA + (1-m1)*sB).mean().item())


METRIC_FUNCS = [
    ("EN",     lambda A, B, F: m_EN(F)),
    ("SF",     lambda A, B, F: m_SF(F)),
    ("SD",     lambda A, B, F: m_SD(F)),
    ("AG",     lambda A, B, F: m_AG(F)),
    ("MI",     lambda A, B, F: m_MI(A, B, F)),
    ("SCD",    lambda A, B, F: m_SCD(A, B, F)),
    ("VIF",    lambda A, B, F: m_VIF(A, B, F)),
    ("Qabf",   lambda A, B, F: m_Qabf(A, B, F)),
    ("MSSSIM", lambda A, B, F: m_MSSSIM(A, B, F)),
    ("SSIMv",  lambda A, B, F: m_SSIMv(A, B, F)),
]


# ── Inference với ghép cặp theo TÊN FILE (an toàn hơn zip thứ tự) ──────────
def list_imgs(d):
    return {f.stem: f for f in sorted(Path(d).iterdir()) if f.suffix.lower() in IMG_EXT}


@torch.no_grad()
def fuse(model, ir_path, vi_path):
    A = cv2.imread(str(ir_path), 0)
    B = cv2.imread(str(vi_path), 0)
    if A is None or B is None:
        return None
    if A.shape != B.shape:
        B = cv2.resize(B, (A.shape[1], A.shape[0]))
    H, W = A.shape
    pad = lambda x: cv2.copyMakeBorder(
        x, 0, (128 - H % 128) % 128, 0, (128 - W % 128) % 128, cv2.BORDER_REFLECT)
    it = torch.from_numpy(pad(A).astype(np.float32))[None, None].cuda()
    vt = torch.from_numpy(pad(B).astype(np.float32))[None, None].cuda()
    out = model(vt, it).cpu().squeeze().numpy()[:H, :W]
    out = np.clip(out, 0, 255)
    if NORM:
        lo, hi = out.min(), out.max()
        if hi - lo >= 1e-6:
            out = (out - lo) / (hi - lo) * 255.0
    return A, B, out.round().clip(0, 255).astype(np.uint8)


def eval_one_dataset(model, ds_dir, save_imgs=None, save_trip=None):
    ir = list_imgs(Path(ds_dir) / "ir")
    vi = list_imgs(Path(ds_dir) / "vi")
    common = sorted(set(ir) & set(vi))
    if not common:
        print(f"  [!] Không có cặp khớp tên trong {ds_dir} (ir∩vi rỗng)."); return None
    rows = []
    for name in common:
        res = fuse(model, ir[name], vi[name])
        if res is None: continue
        A, B, Fu = res
        if save_imgs:
            Path(save_imgs).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(Path(save_imgs) / f"{name}.png"), Fu)
        if save_trip:
            Path(save_trip).mkdir(parents=True, exist_ok=True)
            # IR | VIS | Fused ghép ngang, có vạch trắng ngăn cho dễ nhìn
            sep = np.full((A.shape[0], 4), 255, dtype=np.uint8)
            trip = np.concatenate([A, sep, B, sep, Fu], axis=1)
            cv2.imwrite(str(Path(save_trip) / f"{name}.png"), trip)
        rows.append({k: fn(A, B, Fu) for k, fn in METRIC_FUNCS})
    mean = {k: float(np.mean([r[k] for r in rows])) for k, _ in METRIC_FUNCS}
    std_ = {k: float(np.std([r[k] for r in rows])) for k, _ in METRIC_FUNCS}
    return {"n": len(rows), "mean": mean, "std": std_}


def to_latex(all_results, method_name="ACO-TGFuse"):
    cols = [k for k, _ in METRIC_FUNCS]
    lines = []
    lines.append("% Auto-generated. Một bảng cho mỗi dataset.")
    for ds, r in all_results.items():
        if r is None: continue
        lines.append(f"\n% ── {ds} (n={r['n']}) ──")
        lines.append("\\begin{tabular}{l" + "c" * len(cols) + "}")
        lines.append("\\hline")
        lines.append("Method & " + " & ".join(cols) + " \\\\ \\hline")
        cells = " & ".join(f"{r['mean'][c]:.3f}" for c in cols)
        lines.append(f"{method_name} & {cells} \\\\")
        lines.append("\\hline\n\\end{tabular}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--combo", default="0,3")
    ap.add_argument("--out", default="paper_results")
    ap.add_argument("--save-imgs", action="store_true", help="lưu ảnh fused mỗi dataset")
    ap.add_argument("--triptych", action="store_true",
                    help="lưu thêm ảnh ghép IR|VIS|Fused mỗi dataset")
    ap.add_argument("--norm", action="store_true",
                    help="min-max chuẩn hoá output về [0,255] (công bằng với CDDFuse). "
                         "Nếu bật, phải bật --norm cho eval_external của MỌI SOTA.")
    args = ap.parse_args()
    global NORM
    NORM = args.norm
    if NORM: print("[norm] min-max [0,255] BẬT cho output model.\n")

    os.makedirs(args.out, exist_ok=True)
    model = ACOFusionNet().cuda()
    model.set_branches(tuple(int(x) for x in args.combo.split(",")))
    model.load_state_dict(torch.load(args.model, map_location='cpu'))
    model.eval()
    print(f"Model: {args.model}  combo={args.combo}\n")

    all_results = {}
    for ds in args.datasets:
        ds_name = Path(ds).name
        print(f"── {ds_name} ──")
        save = os.path.join(args.out, "fused", ds_name) if args.save_imgs else None
        trip = os.path.join(args.out, "triptych", ds_name) if args.triptych else None
        r = eval_one_dataset(model, ds, save, trip)
        all_results[ds_name] = r
        if r:
            print(f"  n={r['n']}")
            for k, _ in METRIC_FUNCS:
                print(f"    {k:7s} = {r['mean'][k]:8.4f} ± {r['std'][k]:.3f}")
        print()

    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    with open(os.path.join(args.out, "tables.tex"), "w") as f:
        f.write(to_latex(all_results))
    print(f"Saved → {args.out}/metrics.json và {args.out}/tables.tex")
    print("Dán tables.tex vào bài; thêm dòng cho mỗi SOTA bạn so sánh.")


if __name__ == "__main__":
    main()