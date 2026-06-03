"""
aco_search.py — Ant Colony Optimization tìm KIẾN TRÚC (đường đi) trên supernet.

Khắc phục 2 điểm yếu của ACO cũ:
  1. Search space giờ là 576 cấu hình (không phải 6) → ACO chính danh.
  2. Heuristic η cho node 'pair' được tính TỪ DATA (độ đa dạng feature giữa các
     nhánh, đo bằng 1 - |cosine| của feature pooled), KHÔNG hard-code |i-j|+1 như cũ
     (vốn cài sẵn đáp án (0,3) vào prior). Các node khác η = 1 (trung lập).

Đánh giá ứng viên = reward rẻ tính trong torch trên mini-val (SSIM + SF + SD đã
chuẩn hoá), dùng supernet ĐÃ train one-shot → không retrain từng ứng viên.

Cách dùng:
    python aco_search.py            # cần models_supernet/supernet.model
"""
import os, sys, random, json, math
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import cv2

sys.path.insert(0, '/home/iec/vstung/TGFuse')
from fusion_supernet import FusionSuperNet, SEARCH_SPACE, DECISION_ORDER
from loss import final_ssim


# ── Config ────────────────────────────────────────────────────────────────
class Cfg:
    BASE       = "/home/iec/vstung/TGFuse/dataset_flat"
    supernet   = "/home/iec/vstung/TGFuse/models_supernet/supernet.model"
    H, W       = 256, 256
    n_ants     = 12
    n_iters    = 15
    n_seeds    = 5
    alpha      = 1.0      # mũ pheromone
    beta       = 2.0      # mũ heuristic
    rho        = 0.1      # bay hơi
    top_k      = 3        # số ant tốt nhất được deposit
    val_pairs  = 24
    save       = "/home/iec/vstung/TGFuse/aco_search_results"

cfg = Cfg()


# ── Data ──────────────────────────────────────────────────────────────────
def list_imgs(folder, n=None):
    p = sorted(str(x) for x in Path(folder).iterdir()
               if x.suffix.lower() in {'.jpg', '.png', '.bmp'})
    return p[:n] if n else p


def load_batch(ir_paths, vi_paths, idx):
    def read(paths):
        imgs = []
        for i in idx:
            img = cv2.imread(paths[i], cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (cfg.W, cfg.H))
            imgs.append(img[np.newaxis])
        return torch.from_numpy(np.stack(imgs).astype(np.float32)).cuda()
    return read(vi_paths), read(ir_paths)


# ── Reward rẻ (torch) ──────────────────────────────────────────────────────
def cheap_metrics(ir, vi, fused):
    ssim = final_ssim(ir, vi, fused).item()
    rf = (fused[:, :, :, 1:] - fused[:, :, :, :-1]).pow(2).mean()
    cf = (fused[:, :, 1:, :] - fused[:, :, :-1, :]).pow(2).mean()
    sf = torch.sqrt(rf + cf + 1e-8).item()
    sd = fused.std().item()
    return ssim, sf, sd


# scale thô để chuẩn hoá reward (điều chỉnh theo dải thực tế của bạn nếu cần)
_NORM = {'ssim': (0.6, 0.95), 'sf': (6.0, 14.0), 'sd': (25.0, 60.0)}
def _n(v, lo, hi):
    return max(0.0, min(1.0, (v - lo) / (hi - lo)))


def reward(net, arch, val):
    net.eval()
    ss = sf = sd = 0.0
    with torch.no_grad():
        for vi, ir in val:
            fused = net(vi, ir, arch)
            a, b, c = cheap_metrics(ir, vi, fused)
            ss += a; sf += b; sd += c
    k = len(val)
    ss, sf, sd = ss / k, sf / k, sd / k
    return (_n(ss, *_NORM['ssim']) + _n(sf, *_NORM['sf']) + _n(sd, *_NORM['sd'])) / 3.0


# ── Heuristic η cho 'pair' từ data ─────────────────────────────────────────
def pair_heuristic(net, val):
    """η(pair) = độ đa dạng feature giữa 2 nhánh = 1 - |cosine| (pooled)."""
    feats_acc = [[] for _ in range(4)]
    net.eval()
    with torch.no_grad():
        for vi, ir in val:
            fs = net.encode(vi, ir)
            for k in range(4):
                feats_acc[k].append(F.adaptive_avg_pool2d(fs[k], 1).flatten(1))
    vecs = [torch.cat(a, 0).mean(0) for a in feats_acc]   # [C] mỗi nhánh
    eta = []
    for (i, j) in SEARCH_SPACE['pair']:
        cos = F.cosine_similarity(vecs[i], vecs[j], dim=0).abs().item()
        eta.append(1.0 - cos + 1e-3)
    e = np.array(eta); e = e / e.mean()                   # chuẩn hoá quanh 1
    return e


# ── ACO core ────────────────────────────────────────────────────────────────
def build_eta(net, val):
    eta = {}
    for node in DECISION_ORDER:
        if node == 'pair':
            eta[node] = pair_heuristic(net, val)
        else:
            eta[node] = np.ones(len(SEARCH_SPACE[node]))   # trung lập
    return eta


def sample_arch(tau, eta, rng):
    arch = {}
    for node in DECISION_ORDER:
        p = (tau[node] ** cfg.alpha) * (eta[node] ** cfg.beta)
        p = p / p.sum()
        i = rng.choices(range(len(p)), weights=p, k=1)[0]
        arch[node] = SEARCH_SPACE[node][i]
    return arch


def idx_of(node, val):
    return SEARCH_SPACE[node].index(val)


def run_once(net, val, seed):
    rng = random.Random(seed)
    tau = {node: np.ones(len(SEARCH_SPACE[node])) for node in DECISION_ORDER}
    eta = build_eta(net, val)
    best_arch, best_r = None, -1.0

    for it in range(cfg.n_iters):
        ants = []
        for _ in range(cfg.n_ants):
            arch = sample_arch(tau, eta, rng)
            r = reward(net, arch, val)
            ants.append((r, arch))
            if r > best_r:
                best_r, best_arch = r, dict(arch)
        # bay hơi
        for node in DECISION_ORDER:
            tau[node] *= (1 - cfg.rho)
        # deposit từ top-k
        ants.sort(key=lambda x: x[0], reverse=True)
        for r, arch in ants[:cfg.top_k]:
            for node in DECISION_ORDER:
                tau[node][idx_of(node, arch[node])] += r
        print(f"  seed{seed} iter{it+1:02d}  best_r={best_r:.4f}  {best_arch}")
    return best_arch, best_r


def main():
    os.makedirs(cfg.save, exist_ok=True)
    ir = list_imgs(f"{cfg.BASE}/val/ir")
    vi = list_imgs(f"{cfg.BASE}/val/vi")
    idx_pool = list(range(min(len(ir), len(vi))))
    random.Random(0).shuffle(idx_pool)
    chosen = idx_pool[:cfg.val_pairs]
    val = []
    for s in range(0, len(chosen), 4):
        val.append(load_batch(ir, vi, chosen[s:s+4]))

    net = FusionSuperNet().cuda()
    net.load_state_dict(torch.load(cfg.supernet, map_location='cpu'))
    print(f"Loaded supernet: {cfg.supernet}")

    results = []
    for seed in range(cfg.n_seeds):
        print(f"\n=== ACO seed {seed} ===")
        arch, r = run_once(net, val, seed)
        results.append({'seed': seed, 'reward': r, 'arch': {k: str(v) for k, v in arch.items()}})

    # thống kê: arch xuất hiện nhiều nhất + variance reward
    from collections import Counter
    keys = [tuple(sorted(r['arch'].items())) for r in results]
    most = Counter(keys).most_common(1)[0]
    rewards = [r['reward'] for r in results]
    print("\n" + "=" * 60)
    print(f"Best arch (mode trên {cfg.n_seeds} seeds, xuất hiện {most[1]} lần):")
    print(dict(most[0]))
    print(f"Reward: mean={np.mean(rewards):.4f}  std={np.std(rewards):.4f}")
    print("=" * 60)

    with open(os.path.join(cfg.save, "aco_results.json"), "w") as f:
        json.dump({'per_seed': results,
                   'reward_mean': float(np.mean(rewards)),
                   'reward_std': float(np.std(rewards))}, f, indent=2)
    print(f"Saved → {cfg.save}/aco_results.json")


if __name__ == "__main__":
    main()
