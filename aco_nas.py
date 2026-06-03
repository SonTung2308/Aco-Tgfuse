"""
ACO-NAS: Ant Colony Optimization for Neural Architecture Search
Chọn 2 trong 4 CNN encoder branches tối ưu nhất.

Không gian tìm kiếm: C(4,2) = 6 tổ hợp
  (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
Mỗi "kiến" là một episode training ngắn với 1 tổ hợp.
Reward = SSIM(fused, IR, VIS) trên mini validation set.
Pheromone cập nhật theo reward → hội tụ về tổ hợp tốt nhất.
"""

import numpy as np
import itertools
import torch
from typing import List, Tuple, Dict


class BranchACO:
    """
    Ant Colony Optimization để tìm tổ hợp 2 nhánh tốt nhất
    trong 4 CNN encoder branches.

    Params:
        n_branches   : số nhánh tổng (mặc định 4)
        k_select     : số nhánh cần chọn (mặc định 2)
        alpha        : trọng số pheromone (exploitation)
        beta         : trọng số heuristic (exploration)
        rho          : tốc độ bay hơi pheromone (0..1)
        n_ants       : số kiến mỗi vòng lặp
        tau_init     : pheromone khởi tạo
        tau_min      : pheromone tối thiểu (tránh collapse)
        tau_max      : pheromone tối đa
    """

    def __init__(
        self,
        n_branches: int = 4,
        k_select: int = 2,
        alpha: float = 1.0,
        beta: float = 2.0,
        rho: float = 0.1,
        n_ants: int = 6,
        tau_init: float = 1.0,
        tau_min: float = 0.01,
        tau_max: float = 10.0,
    ):
        self.n_branches = n_branches
        self.k_select = k_select
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.n_ants = n_ants
        self.tau_min = tau_min
        self.tau_max = tau_max

        # Tất cả tổ hợp C(n, k)
        self.combos: List[Tuple[int, ...]] = list(
            itertools.combinations(range(n_branches), k_select)
        )
        self.n_combos = len(self.combos)
        # combo → index
        self.combo_idx: Dict[Tuple[int, ...], int] = {
            c: i for i, c in enumerate(self.combos)
        }

        # Pheromone matrix: mỗi tổ hợp có 1 giá trị
        self.tau = np.full(self.n_combos, tau_init, dtype=float)

        # Heuristic: ưu tiên tổ hợp có khoảng cách scale lớn hơn
        # vì en0 (fine) + en3 (coarse) bổ sung nhau tốt hơn en0+en1
        self.eta = self._compute_heuristic()

        # Lịch sử
        self.history_rewards: List[Dict] = []  # [{combo, reward}, ...]
        self.best_combo: Tuple[int, ...] = self.combos[0]
        self.best_reward: float = 0.0

    def _compute_heuristic(self) -> np.ndarray:
        """
        Heuristic đơn giản: khoảng cách scale giữa 2 nhánh.
        en0=scale0, en1=scale1, en2=scale2, en3=scale3
        Khoảng cách lớn → bổ sung thông tin tốt hơn.
        """
        eta = np.zeros(self.n_combos)
        for i, (a, b) in enumerate(self.combos):
            eta[i] = abs(a - b) + 1.0  # tránh = 0
        return eta / eta.sum() * self.n_combos  # normalize

    def _softmax_probs(self) -> np.ndarray:
        """
        Tính xác suất chọn mỗi tổ hợp dựa trên pheromone + heuristic.
        p(i) = tau(i)^alpha * eta(i)^beta / sum(...)
        """
        scores = (self.tau ** self.alpha) * (self.eta ** self.beta)
        probs = scores / scores.sum()
        return probs

    def select_combo(self, greedy: bool = False) -> Tuple[int, ...]:
        """
        Chọn 1 tổ hợp. greedy=True → chọn combo tốt nhất hiện tại.
        """
        probs = self._softmax_probs()
        if greedy:
            idx = int(np.argmax(probs))
        else:
            idx = np.random.choice(self.n_combos, p=probs)
        return self.combos[idx]

    def select_batch(self, greedy: bool = False) -> List[Tuple[int, ...]]:
        """
        Gửi n_ants kiến, mỗi kiến chọn 1 tổ hợp.
        """
        probs = self._softmax_probs()
        if greedy:
            idxs = [int(np.argmax(probs))] * self.n_ants
        else:
            idxs = np.random.choice(self.n_combos, size=self.n_ants, p=probs)
        return [self.combos[i] for i in idxs]

    def update(self, combo: Tuple[int, ...], reward: float):
        """
        Cập nhật pheromone sau mỗi episode.
        1. Bay hơi tất cả: tau *= (1 - rho)
        2. Cộng reward vào combo được chọn
        3. Clamp trong [tau_min, tau_max]
        """
        # Bay hơi
        self.tau *= (1.0 - self.rho)

        # Deposit
        idx = self.combo_idx[combo]
        self.tau[idx] += reward  # delta tau = reward

        # Clamp
        self.tau = np.clip(self.tau, self.tau_min, self.tau_max)

        # Lịch sử
        self.history_rewards.append({"combo": combo, "reward": reward})

        # Cập nhật best
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_combo = combo

    def update_batch(self, results: List[Dict]):
        """
        Cập nhật từ batch kết quả:
        results = [{"combo": (i,j), "reward": float}, ...]
        """
        # Bay hơi 1 lần
        self.tau *= (1.0 - self.rho)

        for r in results:
            combo = r["combo"]
            reward = r["reward"]
            idx = self.combo_idx[combo]
            self.tau[idx] += reward
            self.history_rewards.append(r)
            if reward > self.best_reward:
                self.best_reward = reward
                self.best_combo = combo

        self.tau = np.clip(self.tau, self.tau_min, self.tau_max)

    def get_best(self) -> Tuple[Tuple[int, ...], float]:
        """Trả về tổ hợp tốt nhất và reward của nó."""
        return self.best_combo, self.best_reward

    def get_probs_table(self) -> str:
        """In bảng xác suất hiện tại để debug."""
        probs = self._softmax_probs()
        lines = [f"{'Combo':<12} {'Tau':>6} {'Eta':>6} {'Prob':>6}"]
        lines.append("-" * 34)
        for i, combo in enumerate(self.combos):
            marker = " <-- best" if combo == self.best_combo else ""
            lines.append(
                f"en{combo[0]}+en{combo[1]}    "
                f"{self.tau[i]:6.3f} {self.eta[i]:6.3f} {probs[i]:6.3f}{marker}"
            )
        return "\n".join(lines)

    def convergence_ratio(self) -> float:
        """
        Tỷ lệ hội tụ: prob của best combo / max có thể.
        = 1.0 nếu tất cả pheromone dồn về 1 combo.
        """
        probs = self._softmax_probs()
        return float(probs.max())

    def save_state(self, path: str):
        np.save(path, {"tau": self.tau, "best_combo": self.best_combo,
                       "best_reward": self.best_reward})

    def load_state(self, path: str):
        state = np.load(path, allow_pickle=True).item()
        self.tau = state["tau"]
        self.best_combo = tuple(state["best_combo"])
        self.best_reward = float(state["best_reward"])
