"""AnalogLinear 测试。

★ 最关键的一条：IDEAL 参数（无量化无噪声）下必须与 nn.Linear 数值一致。
   这是所有模拟层测试的地基 —— 挂了说明基本结构就错了，
   后面所有带噪声的结果都不用看。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.analog import AnalogLinear, analogize, count_tiles, set_chip, set_params
from src.models.autoencoder import DenseAutoEncoder
from src.noise import IDEAL, QUANT_ONLY, DeviceParams


# --------------------------------------------------------------------------
# 退化性质
# --------------------------------------------------------------------------
def test_ideal_params_equal_plain_linear():
    """★ 地基测试。"""
    torch.manual_seed(0)
    al = AnalogLinear(30, 30, IDEAL)
    ref = nn.Linear(30, 30)
    ref.weight.data.copy_(al.weight.data)
    ref.bias.data.copy_(al.bias.data)
    x = torch.randn(8, 30)
    torch.testing.assert_close(al(x), ref(x))


def test_ideal_params_deterministic():
    al = AnalogLinear(30, 30, IDEAL)
    x = torch.randn(4, 30)
    torch.testing.assert_close(al(x), al(x))


def test_quant_only_is_deterministic():
    """只有量化没有噪声时，两次前向必须完全一致。"""
    torch.manual_seed(0)
    al = AnalogLinear(30, 30, QUANT_ONLY)
    x = torch.randn(4, 30)
    torch.testing.assert_close(al(x), al(x))


def test_quant_only_differs_from_ideal():
    torch.manual_seed(0)
    al = AnalogLinear(30, 30, QUANT_ONLY)
    x = torch.randn(16, 30)
    y_q = al(x)
    al.params = IDEAL
    assert not torch.allclose(y_q, al(x))


# --------------------------------------------------------------------------
# D2D vs C2C —— 危险点 A / B
# --------------------------------------------------------------------------
def _d2d_only(sigma: float = 0.05) -> DeviceParams:
    return DeviceParams(w_bits=0, s_bits=0, b_bits=0,
                        sigma_prog=0.0, sigma_d2d=sigma, sigma_read=0.0)


def test_d2d_frozen_across_forwards():
    """★ 危险点 A：固定芯片后，D2D 不得每次前向重采样。"""
    torch.manual_seed(0)
    al = AnalogLinear(30, 30, _d2d_only())
    al.set_chip(0)
    x = torch.randn(4, 30)
    torch.testing.assert_close(al(x), al(x))


def test_d2d_resampled_in_training_mode():
    """chip_id=None 时 D2D 每次前向重采样，让模型对任意芯片鲁棒。"""
    torch.manual_seed(0)
    al = AnalogLinear(30, 30, _d2d_only())
    assert al.chip_id is None
    x = torch.randn(4, 30)
    assert not torch.allclose(al(x), al(x))


def test_same_chip_id_reproducible():
    """同一 chip_id 必须给出同一份 D2D —— 否则跨配置对照会被混杂。"""
    a = AnalogLinear(30, 30, _d2d_only())
    b = AnalogLinear(30, 30, _d2d_only())
    a.set_chip(7)
    b.set_chip(7)
    torch.testing.assert_close(a.d2d, b.d2d)


def test_different_chip_ids_differ():
    al = AnalogLinear(30, 30, _d2d_only())
    al.set_chip(0)
    c0 = al.d2d.clone()
    al.set_chip(1)
    assert not torch.allclose(c0, al.d2d)


def test_d2d_is_buffer_not_parameter():
    """★ 危险点 B：D2D 若注册成 Parameter 会被优化器更新，就不再是固定失配。"""
    al = AnalogLinear(30, 30, _d2d_only())
    assert "d2d" in dict(al.named_buffers())
    assert "d2d" not in dict(al.named_parameters())
    assert not al.d2d.requires_grad


def test_d2d_not_in_state_dict():
    """★ D2D 由 chip_id 派生，不该存进 checkpoint。

    否则 (a) fp32 checkpoint 加载不进模拟模型，(b) 仿真芯片编号会被
    错误地固化进权重文件。
    """
    al = AnalogLinear(30, 30, _d2d_only())
    al.set_chip(3)
    assert "d2d" not in al.state_dict()
    assert "d2d" in dict(al.named_buffers())


def test_fp32_checkpoint_loads_into_analog_model():
    """fp32 训练的 checkpoint 必须能直接加载进模拟模型（C2 组依赖这条）。"""
    torch.manual_seed(0)
    fp = DenseAutoEncoder(input_dim=30, hidden=30, n_blocks=2, bottleneck=4)
    sd = fp.state_dict()
    ana = DenseAutoEncoder(input_dim=30, hidden=30, n_blocks=2, bottleneck=4)
    analogize(ana, DeviceParams())
    ana.load_state_dict(sd)          # 不加 strict=False 也必须成功
    for a, b in zip(ana.parameters(), fp.parameters()):
        torch.testing.assert_close(a, b)


def test_c2c_varies_across_forwards():
    """★ 编程噪声/C2C 必须每次前向重采样。"""
    torch.manual_seed(0)
    p = DeviceParams(w_bits=0, s_bits=0, b_bits=0, sigma_prog=0.05, sigma_d2d=0.0)
    al = AnalogLinear(30, 30, p)
    al.set_chip(0)
    x = torch.randn(4, 30)
    assert not torch.allclose(al(x), al(x))


def test_set_chip_none_restores_unit_d2d():
    al = AnalogLinear(30, 30, _d2d_only())
    al.set_chip(3)
    al.set_chip(None)
    torch.testing.assert_close(al.d2d, torch.ones_like(al.d2d))


# --------------------------------------------------------------------------
# 输出重量化开关（C3 vs C4）
# --------------------------------------------------------------------------
def test_requantize_output_changes_result():
    torch.manual_seed(0)
    x = torch.randn(32, 30)
    c3 = AnalogLinear(30, 30, QUANT_ONLY.with_(requantize_output=False))
    c4 = AnalogLinear(30, 30, QUANT_ONLY.with_(requantize_output=True))
    c4.weight.data.copy_(c3.weight.data)
    c4.bias.data.copy_(c3.bias.data)
    assert not torch.allclose(c3(x), c4(x))


def test_requantize_output_is_noop_when_bits_disabled():
    torch.manual_seed(0)
    p = IDEAL.with_(requantize_output=True)
    al = AnalogLinear(30, 30, p)
    ref = nn.Linear(30, 30)
    ref.weight.data.copy_(al.weight.data)
    ref.bias.data.copy_(al.bias.data)
    x = torch.randn(8, 30)
    torch.testing.assert_close(al(x), ref(x))


# --------------------------------------------------------------------------
# 梯度与模型级工具
# --------------------------------------------------------------------------
def test_gradients_flow_with_noise_and_quant():
    """HWA 训练要求带噪声、带量化时梯度仍然可用。"""
    torch.manual_seed(0)
    al = AnalogLinear(30, 30, DeviceParams())
    x = torch.randn(8, 30, requires_grad=True)
    al(x).sum().backward()
    for t in (x.grad, al.weight.grad, al.bias.grad):
        assert t is not None and torch.isfinite(t).all()


def test_qat_bias_learns_dc_offset():
    """★ 回归（2026-09-22）：量化训练下 bias 必须能学到直流偏移。

    旧 STE 在 Adam 第一步之后让所有 bias 元素幅值相同、全部落在最高电平、梯度为 0，
    bias 从此冻结；输出层只能用权重去合成 log-mel 的直流（约 -26 dB），
    低位宽下权重也被冻结，训练卡在平台上（4 / 5-bit "崩溃"的机制）。
    单层、零均值输入时直流只能来自 bias，所以这里直接检验 bias。
    """
    torch.manual_seed(0)
    al = AnalogLinear(8, 8, QUANT_ONLY)
    target_dc = torch.linspace(-40.0, -10.0, 8)
    opt = torch.optim.Adam(al.parameters(), lr=0.5)
    for _ in range(300):
        x = torch.randn(256, 8)
        loss = ((al(x) - (x + target_dc)) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    # 6-bit、max-abs 尺度下 bias 的 LSB = 40 / 31 ≈ 1.3，量化误差 ≤ 0.65
    assert (al.bias.detach() - target_dc).abs().max() < 1.0, al.bias.detach()


def test_analogize_preserves_weights_and_output():
    torch.manual_seed(0)
    m = DenseAutoEncoder(input_dim=30, hidden=30, n_blocks=2, bottleneck=4).eval()
    x = torch.randn(8, 30)
    before = m(x)
    analogize(m, IDEAL)
    assert any(isinstance(mod, AnalogLinear) for mod in m.modules())
    torch.testing.assert_close(m(x), before)


def test_set_chip_and_params_apply_to_whole_model():
    m = DenseAutoEncoder(input_dim=30, hidden=30, n_blocks=2, bottleneck=4)
    analogize(m, IDEAL)
    set_params(m, DeviceParams())
    set_chip(m, 5)
    layers = [mod for mod in m.modules() if isinstance(mod, AnalogLinear)]
    assert layers and all(l.chip_id == 5 for l in layers)
    assert all(l.params.w_bits == 6 for l in layers)


def test_count_tiles_matches_target_config():
    """目标配置 30 -> [30]x2 -> 4 -> [30]x2 -> 30 应占用 6 个 tile。"""
    m = DenseAutoEncoder(input_dim=30, hidden=30, n_blocks=2, bottleneck=4)
    analogize(m, DeviceParams())
    assert count_tiles(m) == 6


# --------------------------------------------------------------------------
# 层边界 A/D、D/A 放置（Day 4 审稿后新增）
# --------------------------------------------------------------------------
def _target_model(params: DeviceParams):
    return analogize(DenseAutoEncoder(input_dim=30, hidden=30, n_blocks=2, bottleneck=4), params)


def test_adc_mode_none_quantizes_only_model_boundary():
    """★ Ai Linear 架构：只有第一层输入 D/A、最后一层输出 A/D，隐藏层全程模拟。"""
    from src.models.analog import analog_layers
    layers = analog_layers(_target_model(DeviceParams(adc_mode="none")))
    n = len(layers)
    assert [l.quantize_input for l in layers] == [True] + [False] * (n - 1)
    assert [l.quantize_output for l in layers] == [False] * (n - 1) + [True]


def test_adc_mode_per_layer_quantizes_every_boundary():
    from src.models.analog import analog_layers
    layers = analog_layers(_target_model(DeviceParams(adc_mode="per_layer")))
    assert all(l.quantize_input and l.quantize_output for l in layers)


def test_default_adc_mode_reproduces_day3():
    """默认 every_input：每层输入量化、输出看 requantize_output —— 与 Day 3 一致。"""
    from src.models.analog import analog_layers
    layers = analog_layers(_target_model(DeviceParams()))
    assert all(l.quantize_input is True and l.quantize_output is None for l in layers)


def test_hidden_layer_input_stays_analog_in_none_mode():
    """★ 回归测试：Day 3 的缺陷正是隐藏层输入被悄悄量化成 6-bit。
    none 模式下隐藏层必须直接使用连续输入。"""
    from src.models.analog import analog_layers
    from src.models.quant import quantize_ste, quantize_weight_per_tile
    torch.manual_seed(0)
    p = QUANT_ONLY.with_(adc_mode="none")
    hidden = analog_layers(_target_model(p))[1]
    x = torch.randn(64, 30)
    w_q = quantize_weight_per_tile(hidden.weight, p.w_bits, p.tile)
    b_q = quantize_ste(hidden.bias, p.b_bits)
    torch.testing.assert_close(hidden(x), torch.nn.functional.linear(x, w_q, b_q))


def test_first_layer_input_is_quantized_in_none_mode():
    from src.models.analog import analog_layers
    from src.models.quant import quantize_ste, quantize_weight_per_tile
    torch.manual_seed(0)
    p = QUANT_ONLY.with_(adc_mode="none")
    first = analog_layers(_target_model(p))[0]
    x = torch.randn(64, 30)
    w_q = quantize_weight_per_tile(first.weight, p.w_bits, p.tile)
    b_q = quantize_ste(first.bias, p.b_bits)
    assert not torch.allclose(first(x), torch.nn.functional.linear(x, w_q, b_q))


def test_set_params_reapplies_adc_mode():
    from src.models.analog import analog_layers
    m = _target_model(DeviceParams())
    set_params(m, DeviceParams(adc_mode="none"))
    assert analog_layers(m)[1].quantize_input is False


def test_unknown_adc_mode_rejected():
    import pytest
    with pytest.raises(ValueError, match="adc_mode"):
        DeviceParams(adc_mode="sometimes")


def test_read_noise_ignores_bias_magnitude():
    """读噪声按 mean|W·x| 归一，偏置再大也不应放大噪声。"""
    torch.manual_seed(0)
    al = AnalogLinear(30, 30, IDEAL.with_(sigma_read=0.1))
    x = torch.randn(4096, 30)
    lin = torch.nn.functional.linear
    n0 = (al(x) - lin(x, al.weight, al.bias)).std()
    al.bias.data.fill_(100.0)
    n1 = (al(x) - lin(x, al.weight, al.bias)).std()
    assert abs(n1 / n0 - 1) < 0.1


def test_center_removes_dc_before_first_d2a():
    """center=True 时首层 D/A 看到的是去直流后的信号，量程不再被 log-mel 的直流占掉。"""
    torch.manual_seed(0)
    mu = torch.linspace(-40.0, -10.0, 30)
    model = DenseAutoEncoder(30, 30, 2, 4, center=True)
    model.set_input_offset(mu)
    analogize(model, QUANT_ONLY.with_(adc_mode="none"))
    model.eval()
    seen = []
    first = next(m for m in model.modules() if isinstance(m, AnalogLinear))
    first.register_forward_pre_hook(lambda _m, inp: seen.append(inp[0].detach()))
    x = mu + torch.randn(64, 30) * 3.5
    model(x)
    torch.testing.assert_close(seen[0], x - mu)
    assert seen[0].abs().max() < 0.5 * x.abs().max()
