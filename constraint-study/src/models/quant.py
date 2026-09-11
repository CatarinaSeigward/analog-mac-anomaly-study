"""量化：直通估计器（STE）与 LSQ 可学习步长量化。

★ 芯片规格是 **W / S / B 各 6-bit**，三者都要量化。
   常见错误是只量化权重、忘了激活和偏置 —— 那会让结果过于乐观。

约定：``bits <= 0`` 表示**关闭量化**（恒等映射），用于理想对照组与退化测试。

参考
  Esser et al., "Learned Step Size Quantization", ICLR 2020, arXiv:1902.08153
  Nagel et al., "A White Paper on Neural Network Quantization", arXiv:2106.08295
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 直通估计器
# ---------------------------------------------------------------------------
class _RoundSTE(torch.autograd.Function):
    """前向取整，反向直通（梯度原样传递）。"""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return grad_out


def round_ste(x: torch.Tensor) -> torch.Tensor:
    return _RoundSTE.apply(x)


def grad_scale(x: torch.Tensor, scale: float) -> torch.Tensor:
    """前向不变、反向梯度乘以 ``scale``（LSQ 用来稳定步长的梯度）。"""
    return (x - x * scale).detach() + x * scale


def qmax_of(bits: int) -> int:
    """对称量化的正向最大整数电平。6-bit -> 31。"""
    return 2 ** (bits - 1) - 1


# ---------------------------------------------------------------------------
# 无参数量化
# ---------------------------------------------------------------------------
def quantize_ste(x: torch.Tensor, bits: int,
                 scale: torch.Tensor | float | None = None) -> torch.Tensor:
    """对称均匀量化 + STE。

    ``scale`` 为 None 时按 ``x`` 的 max-abs 动态确定（per-tensor）。
    传入张量则按广播规则使用（per-channel / per-tile 由调用方决定）。
    """
    if bits <= 0:
        return x
    qmax = qmax_of(bits)
    if scale is None:
        scale = x.detach().abs().amax().clamp(min=1e-12) / qmax
    return torch.clamp(round_ste(x / scale), -qmax, qmax) * scale


def per_tile_absmax(w: torch.Tensor, tile: int) -> torch.Tensor:
    """逐 tile 的 max-abs，形状与 ``w`` 相同（同一 tile 内的元素共享一个值）。

    ★ 为什么要 per-tile 而不是 per-tensor（ROADMAP 危险点 E）：
      权重被映射到每个 30x30 阵列的电导范围上，尺度是**逐 tile**的。
      用全局尺度的话，绝对值小的 tile 只用到很少的量化电平，
      量化误差会被严重高估，噪声实验的结论就偏了。
    """
    out_f, in_f = w.shape
    # 快速路径：整个矩阵就是一个 tile（目标配置的每一层都是这种情况），
    # pad/view/permute 那套纯属浪费 —— 实测 100us vs 25us，
    # 而 HWA 训练里这个函数每层每步都要调一次。
    if out_f <= tile and in_f <= tile:
        return w.abs().amax().expand(out_f, in_f)

    po, pi = (-out_f) % tile, (-in_f) % tile
    wp = F.pad(w, (0, pi, 0, po))                       # 右下补零，不影响 max-abs
    no, ni = wp.shape[0] // tile, wp.shape[1] // tile
    blocks = wp.view(no, tile, ni, tile).permute(0, 2, 1, 3)   # [no, ni, T, T]
    amax = blocks.abs().amax(dim=(-1, -2))                     # [no, ni]
    full = amax.repeat_interleave(tile, 0).repeat_interleave(tile, 1)
    return full[:out_f, :in_f]


def quantize_weight_per_tile(w: torch.Tensor, bits: int, tile: int) -> torch.Tensor:
    """按 tile 尺度量化权重矩阵 ``[out_features, in_features]``。"""
    if bits <= 0:
        return w
    scale = (per_tile_absmax(w.detach(), tile) / qmax_of(bits)).clamp(min=1e-12)
    return quantize_ste(w, bits, scale)


# ---------------------------------------------------------------------------
# LSQ：可学习步长
# ---------------------------------------------------------------------------
class LSQQuantizer(nn.Module):
    """Learned Step Size Quantization（Esser et al., ICLR 2020）。

    步长 ``s`` 作为可学习参数与权重一起训练，低比特下显著优于固定步长。
    梯度按 ``1/sqrt(numel * qmax)`` 缩放以稳定训练。
    """

    def __init__(self, bits: int, init_from: torch.Tensor | None = None) -> None:
        super().__init__()
        self.bits = bits
        self.register_buffer("initialized", torch.tensor(bits <= 0))
        self.step = nn.Parameter(torch.ones(1))
        if init_from is not None and bits > 0:
            self._init_step(init_from)

    @torch.no_grad()
    def _init_step(self, x: torch.Tensor) -> None:
        """按 LSQ 论文的建议初始化：2 * mean(|x|) / sqrt(qmax)。"""
        s = 2.0 * x.detach().abs().mean() / math.sqrt(max(qmax_of(self.bits), 1))
        self.step.copy_(s.clamp(min=1e-8).reshape(1))
        self.initialized.fill_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bits <= 0:
            return x
        if not bool(self.initialized):
            self._init_step(x)
        qmax = qmax_of(self.bits)
        g = 1.0 / math.sqrt(max(x.numel() * qmax, 1))
        s = grad_scale(self.step.abs().clamp(min=1e-12), g)
        return torch.clamp(round_ste(x / s), -qmax, qmax) * s

    def extra_repr(self) -> str:
        return f"bits={self.bits}"
