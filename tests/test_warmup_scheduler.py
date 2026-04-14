"""elbow-api/training/train_angle_predictor.py の WarmupCosineScheduler テスト."""
import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'elbow-api', 'training'))

import torch
import torch.optim as optim
from train_angle_predictor import WarmupCosineScheduler


def _make_optimizer(lr=0.01):
    """テスト用: 単純なダミーパラメータでAdamを作る."""
    param = torch.nn.Parameter(torch.zeros(4))
    return optim.Adam([param], lr=lr)


class TestWarmupCosineScheduler:
    """WarmupCosineScheduler のテスト."""

    def test_initial_lr_after_first_step(self):
        """最初のstep後: lr = base_lr * (1/warmup_epochs)."""
        base_lr = 0.01
        opt = _make_optimizer(base_lr)
        sched = WarmupCosineScheduler(opt, warmup_epochs=5, total_epochs=20)
        sched.step()
        lr = sched.get_last_lr()[0]
        expected = base_lr * (1 / 5)
        assert abs(lr - expected) < 1e-9

    def test_lr_zero_before_first_step(self):
        """step前: optiomizerの初期LRが保持される."""
        base_lr = 0.01
        opt = _make_optimizer(base_lr)
        sched = WarmupCosineScheduler(opt, warmup_epochs=5, total_epochs=20)
        # step前なのでbase_lrのまま
        assert opt.param_groups[0]["lr"] == base_lr

    def test_warmup_increases_linearly(self):
        """ウォームアップ中はLRが線形に増加する."""
        base_lr = 0.01
        warmup = 5
        opt = _make_optimizer(base_lr)
        sched = WarmupCosineScheduler(opt, warmup_epochs=warmup, total_epochs=30)
        lrs = []
        for _ in range(warmup):
            sched.step()
            lrs.append(sched.get_last_lr()[0])
        # 単調増加していること
        for i in range(1, len(lrs)):
            assert lrs[i] > lrs[i - 1], f"LR should increase: {lrs}"

    def test_warmup_reaches_base_lr(self):
        """ウォームアップ終了時: lr ≈ base_lr."""
        base_lr = 0.01
        warmup = 5
        opt = _make_optimizer(base_lr)
        sched = WarmupCosineScheduler(opt, warmup_epochs=warmup, total_epochs=30)
        for _ in range(warmup):
            sched.step()
        lr = sched.get_last_lr()[0]
        assert abs(lr - base_lr) < 1e-9

    def test_post_warmup_switches_to_cosine(self):
        """ウォームアップ後はCosine Annealingが適用される（LRが下がる）."""
        base_lr = 0.01
        warmup = 3
        total = 15
        opt = _make_optimizer(base_lr)
        sched = WarmupCosineScheduler(opt, warmup_epochs=warmup, total_epochs=total)
        # ウォームアップ完了
        for _ in range(warmup):
            sched.step()
        lr_at_warmup_end = sched.get_last_lr()[0]
        # 数ステップ後: CosineによりLRが減少
        for _ in range(5):
            sched.step()
        lr_after = sched.get_last_lr()[0]
        assert lr_after < lr_at_warmup_end

    def test_get_last_lr_returns_list(self):
        """get_last_lr() はリストを返す."""
        opt = _make_optimizer(0.01)
        sched = WarmupCosineScheduler(opt, warmup_epochs=3, total_epochs=10)
        sched.step()
        result = sched.get_last_lr()
        assert isinstance(result, list)
        assert len(result) == 1

    def test_base_lrs_recorded(self):
        """base_lrs に optimizer の初期LRが記録される."""
        base_lr = 0.005
        opt = _make_optimizer(base_lr)
        sched = WarmupCosineScheduler(opt, warmup_epochs=3, total_epochs=10)
        assert len(sched.base_lrs) == 1
        assert abs(sched.base_lrs[0] - base_lr) < 1e-9

    def test_step_count_increments(self):
        """step呼び出しでステップカウントが増加する."""
        opt = _make_optimizer(0.01)
        sched = WarmupCosineScheduler(opt, warmup_epochs=5, total_epochs=20)
        assert sched._step_count == 0
        sched.step()
        assert sched._step_count == 1
        sched.step()
        assert sched._step_count == 2
