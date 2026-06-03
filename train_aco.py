"""
train_aco.py — Training loop tích hợp ACO-NAS branch selection.

Pipeline:
  Phase 1 (ACO Search, warm_up_epochs):
    - Mỗi episode: ACO chọn 1 tổ hợp (i,j)
    - Train 1 batch nhỏ với tổ hợp đó
    - Tính reward = SSIM trên val mini-batch
    - Cập nhật pheromone
    - Lặp đến khi hội tụ

  Phase 2 (Retrain, retrain_epochs):
    - Dùng tổ hợp tốt nhất từ ACO (cố định)
    - Train đầy đủ giống baseline TGFuse
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
import random
import numpy as np
from os.path import join

from net_aco import ACOFusionNet
from aco_nas import BranchACO
from loss import final_ssim
from function import Vgg16
import utils


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
class Config:
    # Dataset (cấu trúc giống paper: KAIST)
    # Cập nhật đường dẫn phù hợp với máy của bạn
    ir_train_dir  = "/home/iec/vstung/TGFuse/dataset_flat/train/ir"
    vi_train_dir  = "/home/iec/vstung/TGFuse/dataset_flat/train/vi"
    ir_val_dir    = "/home/iec/vstung/TGFuse/dataset_flat/val/ir"     # ~100 ảnh để tính reward
    vi_val_dir    = "/home/iec/vstung/TGFuse/dataset_flat/val/vi"

    train_num     = 40000
    val_num       = 100          # số ảnh để eval reward ACO

    HEIGHT, WIDTH = 256, 256
    batch_size    = 16
    val_batch     = 4            # batch cho val mini-set

    # Training
    lr            = 1e-4
    lr_d          = 1e-4
    warm_up_epochs = 10          # Phase 1: ACO search
    retrain_epochs = 40          # Phase 2: retrain với best arch

    # ACO hyperparams
    aco_alpha     = 1.0
    aco_beta      = 2.0
    aco_rho       = 0.1          # evaporation rate
    aco_n_ants    = 6            # số kiến mỗi round = số combos

    # Save
    save_dir      = "./models_aco"
    log_interval  = 10

cfg = Config()


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def load_images(ir_dir, vi_dir, n, mode='L'):
    """Load n cặp ảnh IR/VIS, trả về list paths."""
    ir_paths = sorted([join(ir_dir, f) for f in os.listdir(ir_dir)
                       if f.lower().endswith(('.png', '.jpg', '.bmp'))])[:n]
    vi_paths = sorted([join(vi_dir, f) for f in os.listdir(vi_dir)
                       if f.lower().endswith(('.png', '.jpg', '.bmp'))])[:n]
    assert len(ir_paths) == len(vi_paths), "IR/VIS count mismatch"
    return ir_paths, vi_paths


def get_batch(ir_paths, vi_paths, indices, height=256, width=256):
    """Load 1 batch từ list paths."""
    ir_list = [ir_paths[i] for i in indices]
    vi_list = [vi_paths[i] for i in indices]
    ir = utils.get_train_images_auto(ir_list, height, width, mode='L').cuda()
    vi = utils.get_train_images_auto(vi_list, height, width, mode='L').cuda()
    return vi, ir


def eval_ssim_reward(model, ir_paths, vi_paths, n_samples=16):
    """
    Tính SSIM reward trên tập val nhỏ.
    Reward = mean SSIM (fused, IR, VIS).
    """
    model.eval()
    ssim_vals = []
    indices = random.sample(range(len(ir_paths)), min(n_samples, len(ir_paths)))

    with torch.no_grad():
        for start in range(0, len(indices), cfg.val_batch):
            batch_idx = indices[start:start + cfg.val_batch]
            vi, ir = get_batch(ir_paths, vi_paths, batch_idx)
            fused = model(vi, ir)
            ssim = final_ssim(ir, vi, fused)
            ssim_vals.append(ssim.item())

    model.train()
    return float(np.mean(ssim_vals))


# ─────────────────────────────────────────────
# Main training
# ─────────────────────────────────────────────
def main():
    os.makedirs(cfg.save_dir, exist_ok=True)

    # Load paths
    ir_train, vi_train = load_images(cfg.ir_train_dir, cfg.vi_train_dir,
                                     cfg.train_num)
    ir_val, vi_val     = load_images(cfg.ir_val_dir, cfg.vi_val_dir,
                                     cfg.val_num)

    n_pairs = len(ir_train)
    batches = n_pairs // cfg.batch_size

    # Models
    gen  = ACOFusionNet().cuda()
    dis1 = Vgg16().cuda()
    dis2 = Vgg16().cuda()

    # ACO
    aco = BranchACO(
        n_branches=4, k_select=2,
        alpha=cfg.aco_alpha, beta=cfg.aco_beta,
        rho=cfg.aco_rho, n_ants=cfg.aco_n_ants,
    )

    L1 = nn.L1Loss()
    writer = SummaryWriter("./log_aco")
    global_step = 0

    print("=" * 60)
    print("PHASE 1: ACO Architecture Search")
    print("=" * 60)

    # ─────────────────────────────────────────────
    # PHASE 1: ACO Search
    # ─────────────────────────────────────────────
    for epoch in range(cfg.warm_up_epochs):
        # Shuffle dữ liệu mỗi epoch
        pair_indices = list(range(n_pairs))
        random.shuffle(pair_indices)

        gen.train()
        episode_results = []

        for batch in range(batches):
            # ACO chọn 1 tổ hợp
            combo = aco.select_combo()
            gen.set_branches(combo)

            # Lấy batch
            idx = pair_indices[batch * cfg.batch_size:(batch + 1) * cfg.batch_size]
            vi, ir = get_batch(ir_train, vi_train, idx)

            # Tối ưu Generator (Var-SSIM)
            opt_G = Adam(gen.parameters(), cfg.lr)
            opt_G.zero_grad()
            fused = gen(vi, ir)
            loss_ssim = 1 - final_ssim(ir, vi, fused)
            loss_ssim.backward()
            opt_G.step()

            # Tối ưu Discriminator VIS (layer 0)
            opt_D1 = Adam(dis1.parameters(), cfg.lr_d)
            opt_D1.zero_grad()
            with torch.no_grad():
                fused_d = gen(vi, ir)
            loss_d1 = L1(dis1(fused_d)[0], dis1(vi)[0])
            loss_d1.backward()
            opt_D1.step()

            # Tối ưu Discriminator IR (layer 2)
            opt_D2 = Adam(dis2.parameters(), cfg.lr_d)
            opt_D2.zero_grad()
            with torch.no_grad():
                fused_d = gen(vi, ir)
            loss_d2 = L1(dis2(fused_d)[2], dis2(ir)[2])
            loss_d2.backward()
            opt_D2.step()

            # Mỗi 20 batch: tính reward và cập nhật pheromone
            if (batch + 1) % 20 == 0:
                reward = eval_ssim_reward(gen, ir_val, vi_val, n_samples=16)
                aco.update(combo, reward)

                if (batch + 1) % (cfg.log_interval * 2) == 0:
                    print(f"  Epoch {epoch+1} batch {batch+1} | "
                          f"combo=en{combo[0]}+en{combo[1]} | "
                          f"reward={reward:.4f} | "
                          f"convergence={aco.convergence_ratio():.3f}")
                    writer.add_scalar("aco/reward", reward, global_step)
                    writer.add_scalar("aco/convergence", aco.convergence_ratio(), global_step)
                    global_step += 1

        # In bảng pheromone mỗi epoch
        print(f"\n[Epoch {epoch+1}] Pheromone table:")
        print(aco.get_probs_table())
        print()

        # Lưu ACO state
        np.save(join(cfg.save_dir, "aco_state.npy"),
                {"tau": aco.tau, "best_combo": aco.best_combo,
                 "best_reward": aco.best_reward})

        # Kiểm tra hội tụ sớm
        if aco.convergence_ratio() > 0.8:
            print(f"[ACO] Converged at epoch {epoch+1}!")
            break

    best_combo, best_reward = aco.get_best()
    print(f"\n[ACO] Best architecture: en{best_combo[0]} + en{best_combo[1]}")
    print(f"[ACO] Best SSIM reward: {best_reward:.4f}")

    # ─────────────────────────────────────────────
    # PHASE 2: Full Retrain với best arch
    # ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"PHASE 2: Retrain với en{best_combo[0]}+en{best_combo[1]}")
    print("=" * 60)

    gen.set_branches(best_combo)

    # Reset optimizers với proper state (fix bug của baseline)
    optimizer_G  = Adam(gen.parameters(),  cfg.lr)
    optimizer_D1 = Adam(dis1.parameters(), cfg.lr_d)
    optimizer_D2 = Adam(dis2.parameters(), cfg.lr_d)

    tbar = trange(cfg.retrain_epochs, ncols=120)
    all_ssim, all_d1, all_d2 = 0.0, 0.0, 0.0
    w_num = 0

    for epoch in tbar:
        pair_indices = list(range(n_pairs))
        random.shuffle(pair_indices)
        gen.train()

        for batch in range(batches):
            idx = pair_indices[batch * cfg.batch_size:(batch + 1) * cfg.batch_size]
            vi, ir = get_batch(ir_train, vi_train, idx)

            # Generator
            optimizer_G.zero_grad()
            fused = gen(vi, ir)
            loss_g = 1 - final_ssim(ir, vi, fused)
            loss_g.backward()
            optimizer_G.step()

            # Dis1 (VIS)
            optimizer_D1.zero_grad()
            with torch.no_grad():
                fused_d = gen(vi, ir)
            loss_d1 = L1(dis1(fused_d)[0], dis1(vi)[0])
            loss_d1.backward()
            optimizer_D1.step()

            # Dis2 (IR)
            optimizer_D2.zero_grad()
            with torch.no_grad():
                fused_d = gen(vi, ir)
            loss_d2 = L1(dis2(fused_d)[2], dis2(ir)[2])
            loss_d2.backward()
            optimizer_D2.step()

            all_ssim += loss_g.item()
            all_d1   += loss_d1.item()
            all_d2   += loss_d2.item()

            if (batch + 1) % cfg.log_interval == 0:
                msg = (f"E{epoch+1} [{batch+1}/{batches}] "
                       f"ssim={all_ssim/cfg.log_interval:.4f} "
                       f"d1={all_d1/cfg.log_interval:.4f} "
                       f"d2={all_d2/cfg.log_interval:.4f}")
                tbar.set_description(msg)
                writer.add_scalar("retrain/ssim_loss", all_ssim / cfg.log_interval, w_num)
                writer.add_scalar("retrain/dis1_loss", all_d1 / cfg.log_interval, w_num)
                writer.add_scalar("retrain/dis2_loss", all_d2 / cfg.log_interval, w_num)
                w_num += 1
                all_ssim = all_d1 = all_d2 = 0.0

        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            ckpt = join(cfg.save_dir,
                        f"ACO_en{best_combo[0]}en{best_combo[1]}_epoch{epoch+1}.model")
            gen.eval().cpu()
            torch.save(gen.state_dict(), ckpt)
            gen.train().cuda()
            print(f"\n[Saved] {ckpt}")

    # Final save
    final_path = join(cfg.save_dir,
                      f"ACO_en{best_combo[0]}en{best_combo[1]}_final.model")
    gen.eval().cpu()
    torch.save(gen.state_dict(), final_path)
    print(f"\n[Done] Final model: {final_path}")
    writer.close()


if __name__ == "__main__":
    main()