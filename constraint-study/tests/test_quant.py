"""量化模块测试。"""

from __future__ import annotations

import torch

from src.models.quant import (
    LSQQuantizer,
    per_tile_absmax,
    qmax_of,
    quantize_ste,
    quantize_weight_per_tile,
    round_ste,
)


# --------------------------------------------------------------------------
# STE
# --------------------------------------------------------------------------
def test_round_ste_forward_is_round():
    x = torch.tensor([-1.6, -0.4, 0.4, 1.5, 2.5])
    torch.testing.assert_close(round_ste(x), torch.round(x))


def test_ste_gradient_passes_through():
    """★ 量化区间**内**的梯度必须直通，否则量化后的网络根本训不动。

    显式 scale 留了一倍余量。默认 max-abs 尺度（最大元素恰好在最高电平）由
    test_ste_gradient_passes_at_top_level_with_absmax_scale 覆盖。
    ⚠️ 旧注释曾把"最大元素梯度被置零"当作正确行为 —— 实际被置零的是整个最高电平带，
       那是 bug（见 quant._RoundClampSTE）。
    """
    x = torch.randn(64, requires_grad=True)
    scale = x.detach().abs().amax() / (qmax_of(6) / 2)   # 留一倍余量，无元素触边
    quantize_ste(x, bits=6, scale=scale).sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))


def test_ste_gradient_passes_at_top_level_with_absmax_scale():
    """★ 回归（2026-09-22）：取整到最高电平 ±qmax 的元素梯度必须直通。

    旧实现先取整再 clamp，而 torch 的 clamp 在恰好等于边界时梯度为 0 ——
    于是整个最高电平带（|x/s| ∈ [qmax-0.5, qmax]）都没有梯度，默认 max-abs
    尺度下最大的元素永远在这个带里。max-abs 尺度下没有任何元素被裁剪，梯度应全部直通。
    """
    x = torch.tensor([0.10, 0.20, -0.30, 0.31, 0.61, 0.62], requires_grad=True)
    quantize_ste(x, bits=6).sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))


def test_ste_gradient_passes_when_all_magnitudes_equal():
    """★ 回归：所有元素幅值相同时（如 Adam 第一步之后从零初始化的 bias），
    旧实现整个张量梯度为 0，参数从此冻结。"""
    x = torch.full((30,), 0.002)
    x[::2] *= -1
    x.requires_grad_(True)
    quantize_ste(x, bits=6).sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))


def test_lsq_input_gradient_passes_at_top_level():
    """LSQ 同样的结构问题：落在最高电平但未越界的输入，梯度必须直通。"""
    q = LSQQuantizer(bits=4)
    q.step.data.fill_(0.1)
    q.initialized.fill_(True)
    x = torch.tensor([0.0, 0.3, 0.68, -0.70], requires_grad=True)   # 0.68 / 0.1 -> 电平 7 = qmax
    q(x).sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))


def test_ste_gradient_zero_outside_range():
    """超出量化范围的值梯度应为 0 —— 这正是裁剪阈值有意义的原因。"""
    x = torch.tensor([0.0, 1.0, 100.0, -100.0], requires_grad=True)
    quantize_ste(x, bits=6, scale=torch.tensor(1.0)).sum().backward()
    torch.testing.assert_close(x.grad, torch.tensor([1.0, 1.0, 0.0, 0.0]))


def test_quantize_disabled_when_bits_le_zero():
    x = torch.randn(32)
    for bits in (0, -1):
        torch.testing.assert_close(quantize_ste(x, bits), x)


# --------------------------------------------------------------------------
# 量化电平
# --------------------------------------------------------------------------
def test_qmax():
    assert qmax_of(6) == 31
    assert qmax_of(8) == 127
    assert qmax_of(1) == 0


def test_quantization_level_count():
    """6-bit 对称量化的不同取值数不应超过 2*qmax+1 = 63。"""
    torch.manual_seed(0)
    x = torch.randn(10_000)
    q = quantize_ste(x, bits=6)
    assert len(torch.unique(q)) <= 2 * qmax_of(6) + 1


def test_quantization_error_shrinks_with_bits():
    torch.manual_seed(0)
    x = torch.randn(4096)
    errs = [(quantize_ste(x, b) - x).abs().mean().item() for b in (4, 6, 8, 10)]
    assert errs == sorted(errs, reverse=True), errs


def test_quantize_is_symmetric_around_zero():
    x = torch.randn(1000)
    torch.testing.assert_close(quantize_ste(-x, 6), -quantize_ste(x, 6))


# --------------------------------------------------------------------------
# per-tile 尺度
# --------------------------------------------------------------------------
def test_per_tile_absmax_values():
    """构造两个 tile，各自最大值不同，验证尺度不串台。"""
    w = torch.zeros(4, 8)
    w[0, 0] = 3.0     # 左上 tile
    w[0, 4] = -7.0    # 右上 tile
    amax = per_tile_absmax(w, tile=4)
    assert amax.shape == w.shape
    assert torch.all(amax[:, :4] == 3.0)
    assert torch.all(amax[:, 4:] == 7.0)


def test_per_tile_absmax_handles_ragged_shapes():
    """out/in 不是 tile 整数倍时，补零不应影响 max-abs。"""
    torch.manual_seed(0)
    w = torch.randn(31, 61)
    amax = per_tile_absmax(w, tile=30)
    assert amax.shape == w.shape
    assert torch.all(amax >= w.abs())


def test_per_tile_beats_per_tensor_on_uneven_weights():
    """★ 危险点 E：某个 tile 幅值远小于其他 tile 时，
       per-tensor 尺度会让它几乎没有量化电平可用。"""
    torch.manual_seed(0)
    w = torch.randn(60, 60) * 0.001
    w[:30, :30] *= 1000.0                       # 左上 tile 幅值大 1000 倍
    err_tile = (quantize_weight_per_tile(w, 6, 30) - w).abs().mean()
    err_tensor = (quantize_ste(w, 6) - w).abs().mean()
    assert err_tile < err_tensor, (err_tile.item(), err_tensor.item())


# --------------------------------------------------------------------------
# LSQ
# --------------------------------------------------------------------------
def test_lsq_step_is_learnable():
    q = LSQQuantizer(bits=6)
    x = torch.randn(256, requires_grad=True)
    q(x).pow(2).sum().backward()
    assert q.step.grad is not None
    assert torch.isfinite(q.step.grad).all()


def test_lsq_disabled_when_bits_le_zero():
    q = LSQQuantizer(bits=0)
    x = torch.randn(16)
    torch.testing.assert_close(q(x), x)


def test_lsq_output_is_quantized():
    torch.manual_seed(0)
    q = LSQQuantizer(bits=4)
    out = q(torch.randn(5000))
    assert len(torch.unique(out)) <= 2 * qmax_of(4) + 1
