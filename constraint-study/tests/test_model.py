"""模型结构测试。"""

from __future__ import annotations

import torch

from src.models.autoencoder import DenseAutoEncoder, reconstruction_error
from src.utils import count_params


def test_baseline_shapes():
    """C0 官方基线：640 -> [128]x4 -> 8 -> [128]x4 -> 640。"""
    m = DenseAutoEncoder(input_dim=640, hidden=128, n_blocks=4, bottleneck=8)
    x = torch.randn(16, 640)
    assert m(x).shape == (16, 640)
    assert m.encoder(x).shape == (16, 8)  # 瓶颈维度


def test_squeezed_shapes():
    """Day 2 目标配置：30 -> [30]x2 -> 4 -> [30]x2 -> 30。"""
    m = DenseAutoEncoder(input_dim=30, hidden=30, n_blocks=2, bottleneck=4)
    x = torch.randn(8, 30)
    assert m(x).shape == (8, 30)
    # 压缩后参数量应比基线小一个数量级以上
    # 实测：基线 267,928 -> 压缩后 4,242，约 1/63
    base = count_params(DenseAutoEncoder(640, 128, 4, 8))
    assert count_params(m) < base / 50, f"{count_params(m)} vs base {base}"
    assert count_params(m) < 10_000


def test_reconstruction_error_shape_and_sign():
    m = DenseAutoEncoder(input_dim=32, hidden=16, n_blocks=1, bottleneck=4)
    m.eval()
    x = torch.randn(5, 32)
    err = reconstruction_error(m, x)
    assert err.shape == (5,)
    assert (err >= 0).all()


def test_perfect_reconstruction_gives_zero_error():
    """恒等映射下重构误差应为 0 —— 验证误差是逐样本按特征维取均值的。"""

    class Identity(torch.nn.Module):
        def forward(self, x):
            return x

    x = torch.randn(4, 10)
    assert torch.allclose(reconstruction_error(Identity(), x), torch.zeros(4))


def test_center_is_a_reparametrisation():
    """center=True 只是把输入和重构目标平移同一个常数：AE_c(x) = AE(x - μ) + μ。"""
    torch.manual_seed(0)
    mu = torch.linspace(-40.0, -10.0, 30)
    m_c = DenseAutoEncoder(30, 30, 2, 4, center=True)
    m_c.set_input_offset(mu)
    m = DenseAutoEncoder(30, 30, 2, 4)
    m.load_state_dict({k: v for k, v in m_c.state_dict().items() if k != "input_offset"})
    m_c.eval()
    m.eval()
    x = mu + torch.randn(16, 30) * 3.5
    torch.testing.assert_close(m_c(x), m(x - mu) + mu)
    # 重构误差（异常分数）与在去直流坐标下计算的完全一致
    torch.testing.assert_close(reconstruction_error(m_c, x), reconstruction_error(m, x - mu))


def test_center_offset_roundtrips_through_state_dict():
    mu = torch.linspace(-40.0, -10.0, 30)
    m = DenseAutoEncoder(30, 30, 2, 4, center=True)
    m.set_input_offset(mu)
    m2 = DenseAutoEncoder(30, 30, 2, 4, center=True)
    m2.load_state_dict(m.state_dict())
    torch.testing.assert_close(m2.input_offset, mu)
    # 旧 checkpoint（无 input_offset）仍能加载进默认模型
    assert "input_offset" not in DenseAutoEncoder(30, 30, 2, 4).state_dict()
