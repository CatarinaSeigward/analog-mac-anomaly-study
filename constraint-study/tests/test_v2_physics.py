"""v2 物理模型测试：固定满量程、饱和激活、三种读噪声模型（2026-09-22）。

v1 路径（默认参数）必须与 v2 改动前逐位一致 —— 那条由改动前后的逐位比对保证，
这里只测新行为。
"""

from __future__ import annotations

import pytest
import torch

from src.models.analog import AnalogLinear, analog_layers, analogize, set_chip
from src.models.autoencoder import ACTIVATIONS, DenseAutoEncoder
from src.noise import DeviceParams

FIXED = DeviceParams(sigma_prog=0.0, sigma_d2d=0.0, sigma_read=0.0, scale_mode="fixed")


def _train_forward(layer: AnalogLinear, x: torch.Tensor) -> torch.Tensor:
    layer.train()
    with torch.enable_grad():
        return layer(x)


# --------------------------------------------------------------------------
# 固定满量程（A16）
# --------------------------------------------------------------------------
def test_fixed_scale_calibrates_in_training_and_freezes_in_eval():
    torch.manual_seed(0)
    al = AnalogLinear(30, 30, FIXED)
    x = torch.randn(4096, 30)
    _train_forward(al, x)
    fs = float(al.in_fs)
    assert fs == pytest.approx(float(torch.quantile(x.abs().flatten(), 0.999)), rel=1e-5)
    al.eval()
    al(x * 10.0)                                   # 评估不再改满量程
    assert float(al.in_fs) == fs


def test_fixed_scale_clips_beyond_full_scale():
    torch.manual_seed(0)
    al = AnalogLinear(4, 4, FIXED)
    _train_forward(al, torch.randn(4096, 4))
    fs = float(al.in_fs)
    seen = []
    al.register_forward_hook(lambda _m, inp, _o: seen.append(inp[0]))
    al.eval()
    x = torch.tensor([[100.0, -100.0, 0.0, 0.5]])
    from src.models.quant import quantize_ste
    xq = quantize_ste(x, 6, al.in_fs / 31)
    assert float(xq.abs().max()) == pytest.approx(fs, rel=1e-5)


def test_fixed_scale_is_independent_of_batch_composition():
    """★ A16：固定满量程下，一个样本的 D/A 结果与同 batch 的其他样本无关；动态 max-abs 下有关。"""
    torch.manual_seed(0)
    for mode, expect_equal in (("fixed", True), ("dynamic", False)):
        al = AnalogLinear(30, 30, FIXED.with_(scale_mode=mode))
        _train_forward(al, torch.randn(4096, 30))
        al.eval()
        x = torch.randn(8, 30)
        loud = torch.cat([x, 50.0 * torch.ones(1, 30)])      # 同 batch 里多一个很响的样本
        same = torch.allclose(al(x), al(loud)[:8])
        assert same == expect_equal, mode


def test_fixed_scale_requires_calibration():
    al = AnalogLinear(30, 30, FIXED)
    al.eval()
    with pytest.raises(RuntimeError, match="未标定"):
        al(torch.randn(4, 30))


def test_range_tracking_skipped_without_grad():
    """BN 重标定在 no_grad 下把模型切到 train 模式 —— 不能顺手改掉满量程。"""
    torch.manual_seed(0)
    al = AnalogLinear(30, 30, FIXED)
    _train_forward(al, torch.randn(4096, 30))
    fs = float(al.in_fs)
    al.train()
    with torch.no_grad():
        al(torch.randn(4096, 30) * 10.0)
    assert float(al.in_fs) == fs


def test_v1_path_does_not_track_ranges():
    al = AnalogLinear(30, 30, DeviceParams())
    _train_forward(al, torch.randn(256, 30))
    assert float(al.in_fs) == 0.0 and float(al.out_fs) == 0.0


def test_checkpoint_without_fs_buffers_loads():
    """v2 之前的 checkpoint 没有 in_fs / out_fs，按"未标定"加载，不报错。"""
    al = AnalogLinear(30, 30, FIXED)
    sd = {k: v for k, v in al.state_dict().items() if k not in ("in_fs", "out_fs")}
    AnalogLinear(30, 30, FIXED).load_state_dict(sd)


def test_fs_buffers_roundtrip_through_state_dict():
    torch.manual_seed(0)
    al = AnalogLinear(30, 30, FIXED)
    _train_forward(al, torch.randn(4096, 30))
    al2 = AnalogLinear(30, 30, FIXED)
    al2.load_state_dict(al.state_dict())
    assert float(al2.in_fs) == float(al.in_fs) > 0


# --------------------------------------------------------------------------
# 读噪声模型
# --------------------------------------------------------------------------
def _noise_std(noise_model: str, x: torch.Tensor, in_features: int = 30, reps: int = 200):
    """同一输入重复前向，返回逐元素噪声 std 与无噪声的信号 |W·x|。"""
    torch.manual_seed(0)
    p = DeviceParams(w_bits=0, s_bits=0, b_bits=0, sigma_prog=0.0, sigma_d2d=0.0,
                     sigma_read=0.1, noise_model=noise_model)
    al = AnalogLinear(in_features, 30, p, bias=False)
    al.in_range_fixed = 1.0                        # 单元满量程 U = max|w| × 1
    al.eval()
    with torch.no_grad():
        clean = x @ al.weight.T
        ys = torch.stack([al(x) for _ in range(reps)])
        return (ys - clean).std(0), clean.abs(), al.cell_full_scale()


def test_absolute_noise_is_independent_of_signal():
    x = torch.randn(64, 30)
    std_small, _, u = _noise_std("absolute", 0.1 * x)
    std_big, _, _ = _noise_std("absolute", 10.0 * x)
    assert std_small.mean() == pytest.approx(0.1 * float(u), rel=0.05)
    assert std_big.mean() == pytest.approx(std_small.mean(), rel=0.05)


def test_proportional_noise_scales_with_signal():
    x = torch.randn(64, 30)
    std, sig, _ = _noise_std("proportional", x)
    ratio = std / sig.clamp(min=1e-6)
    keep = sig > 0.1
    assert ratio[keep].mean() == pytest.approx(0.1, rel=0.05)


def test_shot_noise_scales_with_sqrt_signal():
    x = torch.randn(64, 30)
    std, sig, u = _noise_std("shot", x)
    keep = sig > 0.1
    expected = 0.1 * torch.sqrt(u * sig[keep])
    assert (std[keep] / expected).mean() == pytest.approx(1.0, rel=0.05)


def test_absolute_noise_adds_per_input_tile():
    """输入方向 2 个 tile 的部分和相加：各自噪声独立，std 乘 √2 —— 多 tile 的代价。"""
    std30, _, u30 = _noise_std("absolute", torch.randn(64, 30), in_features=30)
    std60, _, u60 = _noise_std("absolute", torch.randn(64, 60), in_features=60)
    assert (std60.mean() / u60) / (std30.mean() / u30) == pytest.approx(2 ** 0.5, rel=0.05)


def test_unknown_noise_model_rejected():
    with pytest.raises(ValueError):
        DeviceParams(noise_model="thermal")


# --------------------------------------------------------------------------
# 饱和激活与输入范围（A4）
# --------------------------------------------------------------------------
@pytest.mark.parametrize("act", ACTIVATIONS)
def test_activation_ranges(act):
    torch.manual_seed(0)
    m = DenseAutoEncoder(30, 30, 2, 4, act=act).eval()
    acts = []
    for mod in m.modules():
        if isinstance(mod, (torch.nn.ReLU, torch.nn.Hardtanh, torch.nn.Tanh)):
            mod.register_forward_hook(lambda _m, _i, o: acts.append(o))
    with torch.no_grad():
        m(torch.randn(256, 30) * 10)
    lo = min(float(a.min()) for a in acts)
    hi = max(float(a.max()) for a in acts)
    if act == "relu":
        assert lo >= 0.0
    elif act == "clipped_relu":
        assert lo >= 0.0 and hi <= 1.0
    else:
        assert lo >= -1.0 and hi <= 1.0


def test_signal_ranges_follow_the_activation():
    """饱和激活之后的隐藏层输入范围 = 饱和轨；首层来自 D/A；ReLU 不饱和，不设。"""
    for act, rail in (("clipped_relu", 1.0), ("tanh", 1.0), ("relu", None)):
        m = DenseAutoEncoder(30, 30, 2, 4, act=act)
        analogize(m, DeviceParams(adc_mode="none"))
        ranges = [layer.in_range_fixed for layer in analog_layers(m)]
        assert ranges[0] is None
        assert all(r == rail for r in ranges[1:]), (act, ranges)


def test_absolute_noise_cannot_be_escaped_by_shrinking_the_signal():
    """★ 物理要点：饱和激活下，隐藏层的单元满量程由饱和轨决定，与网络实际用了多少量程无关。
    网络把信号整体缩小，absolute 噪声不跟着缩小（v1 的噪声会）。"""
    torch.manual_seed(0)
    p = DeviceParams(sigma_prog=0.0, sigma_d2d=0.0, sigma_read=0.1, noise_model="absolute",
                     scale_mode="fixed", adc_mode="none")
    m = DenseAutoEncoder(30, 30, 2, 4, act="clipped_relu")
    analogize(m, p)
    m.train()
    with torch.enable_grad():
        m(torch.randn(512, 30))                     # 标定首层满量程
    hidden = analog_layers(m)[1]
    u_before = float(hidden.cell_full_scale())
    set_chip(m, None)
    assert hidden.in_range_fixed == 1.0
    assert float(hidden.cell_full_scale()) == u_before   # 与输入信号大小无关，只看 max|w| × 饱和轨
