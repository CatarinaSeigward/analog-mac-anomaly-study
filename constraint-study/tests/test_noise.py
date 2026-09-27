"""噪声参数与随机源测试。

重点验证 ROADMAP 危险点 C：芯片采样的随机源必须独立于训练随机源，
否则"芯片编号"与"训练种子"混杂，跨芯片方差的归因就是错的。
"""

from __future__ import annotations

import torch

from src.noise import (
    CHIP_SEED_BASE,
    IDEAL,
    QUANT_ONLY,
    DeviceParams,
    chip_generator,
    sample_d2d,
)


# --------------------------------------------------------------------------
# DeviceParams
# --------------------------------------------------------------------------
def test_defaults_match_chip_spec():
    """芯片规格：W/S/B 各 6-bit，30x30 阵列，层间无 ADC。"""
    p = DeviceParams()
    assert (p.w_bits, p.s_bits, p.b_bits) == (6, 6, 6)
    assert p.tile == 30
    assert p.requantize_output is False


def test_presets():
    assert IDEAL.noise_free and IDEAL.w_bits <= 0
    assert QUANT_ONLY.noise_free and QUANT_ONLY.w_bits == 6


def test_with_returns_new_instance():
    p = DeviceParams()
    q = p.with_(sigma_prog=0.10)
    assert q.sigma_prog == 0.10
    assert p.sigma_prog == 0.03      # 原实例不变（frozen）
    assert q.w_bits == p.w_bits


# --------------------------------------------------------------------------
# 芯片采样
# --------------------------------------------------------------------------
def test_same_chip_id_gives_identical_d2d():
    a = sample_d2d((30, 30), 0.05, chip_generator(3))
    b = sample_d2d((30, 30), 0.05, chip_generator(3))
    torch.testing.assert_close(a, b)


def test_different_chip_ids_give_different_d2d():
    a = sample_d2d((30, 30), 0.05, chip_generator(0))
    b = sample_d2d((30, 30), 0.05, chip_generator(1))
    assert not torch.allclose(a, b)


def test_chip_sampling_independent_of_global_seed():
    """★ 危险点 C：全局训练种子变化不得影响芯片采样。"""
    torch.manual_seed(0)
    a = sample_d2d((30, 30), 0.05, chip_generator(2))
    torch.manual_seed(12345)
    b = sample_d2d((30, 30), 0.05, chip_generator(2))
    torch.testing.assert_close(a, b)


def test_chip_sampling_does_not_disturb_global_rng():
    """芯片采样不应消耗全局 RNG，否则训练可复现性会被芯片数影响。"""
    torch.manual_seed(0)
    before = torch.randn(4)
    torch.manual_seed(0)
    sample_d2d((30, 30), 0.05, chip_generator(9))
    after = torch.randn(4)
    torch.testing.assert_close(before, after)


def test_zero_sigma_gives_unit_factor():
    d = sample_d2d((8, 8), 0.0, chip_generator(0))
    torch.testing.assert_close(d, torch.ones(8, 8))


def test_d2d_statistics():
    """均值应接近 1，标准差应接近 sigma_d2d。"""
    d = sample_d2d((400, 400), 0.05, chip_generator(0))
    assert abs(d.mean().item() - 1.0) < 0.01
    assert abs(d.std().item() - 0.05) < 0.005


def test_chip_generator_seed_offset():
    g = chip_generator(5)
    assert g.initial_seed() == CHIP_SEED_BASE + 5
