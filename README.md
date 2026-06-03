# ACO-TGFuse — Bộ cải tiến 
## File nào làm gì

| File | Vai trò |
|------|---------|
| `aco_modules.py` | Building blocks dùng chung: `SpatialGate` (per-pixel), `CrossBranchAttention` (bản viết lại sạch), `AdaptiveScaleWeights`, `EdgeResidual` |
| `losses_v2.py` | `saliency_intensity_loss`, `final_ssim_soft`, `gradient_loss`, `edge_loss`, `UncertaintyWeighter` (Kendall) |
| `net_aco_v5.py` | Mạng **production** cho cấu hình tốt (mặc định en0+en3) — đã gắn cross-attn + spatial gate + adaptive merge + edge |
| `fusion_supernet.py` | **Supernet** weight-sharing, forward route theo `arch`. Search space = 576 cấu hình |
| `train_supernet.py` | Train supernet one-shot (single-path uniform) — chạy TRƯỚC khi search |
| `aco_search.py` | **ACO path-search** trên supernet + heuristic từ data + multi-seed variance |
| `train_v9.py` | Fine-tune `net_aco_v5` với loss mới + checkpoint theo composite/Pareto |

## Thứ tự chạy

**Nhánh A — chỉ muốn tăng số nhanh (không cần NAS):**
```bash
python train_v9.py        # resume từ models_aco_v5/best.model, dùng loss mới
```
Chọn checkpoint trong `models_aco_v9/`: `best_composite.model` (cân bằng),
hoặc `best_sf.model` / `best_sd.model` / `best_ssim.model` tùy metric ưu tiên.

**Nhánh B — làm ACO-NAS cho đàng hoàng (cho phần đóng góp của bài):**
```bash
python train_supernet.py  # → models_supernet/supernet.model  (~15 epoch)
python aco_search.py      # → aco_search_results/aco_results.json
# Lấy arch tốt nhất, set vào net_aco_v5/fusion_supernet rồi fine-tune lại bằng train_v9
```

## PHẢI tinh chỉnh trước khi tin số

1. **Dải chuẩn hoá reward** trong `aco_search.py::_NORM` và `train_v9.py::composite`
   đang đặt thô (ssim 0.6–0.95, sf 6–14, sd 25–60). Đo dải thực trên val của bạn
   rồi chỉnh lại, nếu không composite/reward sẽ lệch.

2. **Bug SD = 35 (vs paper 94)**: trước khi so sánh, kiểm tra pipeline đánh giá
   `evaluate_final.compute_all` — SD chỉ là std pixel, không nên lệch 3× nếu output
   đúng. Rất có thể output đang bị nén contrast hoặc bạn tính SD trên ảnh [0,1].
   Dùng bộ metric chuẩn ngành (VIFB/MEFB) để số so được với paper.

3. **Heuristic η của 'pair'** giờ tính từ độ đa dạng feature (1−|cosine|), KHÔNG
   hard-code `|i-j|+1`. Nếu ACO vẫn ra (0,3) từ η trung lập này thì kết luận mới
   có giá trị khoa học. Chạy đủ `n_seeds` và report mean±std.

## Lưu ý kỹ thuật

- `ConvLayer` mới KHÔNG ép `.cuda()` bên trong. Nhớ `model.cuda()` + input `.cuda()`
  (các train script đã làm sẵn).
- `UncertaintyWeighter` có tham số học được → đã được đưa vào `opt_G`. Đừng quên
  nếu bạn viết train script riêng.
- Cross-attention cũ (net_aco_v4) hỏng do code rối, không phải do ý tưởng. Bản mới
  trong `aco_modules.CrossBranchAttention` là residual+LN chuẩn, test riêng được.
