"""全连接自编码器 —— DCASE2020 / MLPerf Tiny 异常检测基线。

结构（默认参数即官方基线）::

    input(640)
      -> [Linear(128) + BN + ReLU] x4        编码器
      -> [Linear(8)   + BN + ReLU]           瓶颈
      -> [Linear(128) + BN + ReLU] x4        解码器
      -> Linear(640)                          输出（线性）

★ 全部是全连接层，没有卷积 —— 这正是它适合模拟 MAC 阵列的原因，
   也是本研究选它作起点的理由（见 PLAN 附录 A）。

宽度 / 深度 / 瓶颈都参数化，Day 2 的压缩实验直接改配置即可：
    baseline : hidden=128, n_blocks=4, bottleneck=8, input_dim=640
    压缩后   : hidden=30,  n_blocks=2, bottleneck=4, input_dim=30
"""

from __future__ import annotations

import torch
import torch.nn as nn


ACTIVATIONS = ("relu", "clipped_relu", "tanh")
"""隐藏层非线性（assumptions A4）。

``relu``          v1，不饱和
``clipped_relu``  min(max(z, 0), 1)：单极性电流镜天然整流，上限是偏置电流 —— current-mode 最自然的形状
``tanh``          差分对的形状，(-1, 1)

饱和轨取 1：激活前有 BN，其增益 γ、偏置 β 可学习，饱和点改成 k 倍，网络把 γ、β 同步放大
k 倍即可完全抵消。所以饱和点的绝对位置没有意义，有意义的是信号通路的**动态范围**
（饱和轨 / 噪声底），它由 noise_model="absolute" 的 σ 决定（见 src/noise.py）。
"""
ACT_RAIL = {"relu": None, "clipped_relu": 1.0, "tanh": 1.0}


def _activation(act: str) -> nn.Module:
    if act == "relu":
        return nn.ReLU(inplace=True)
    if act == "clipped_relu":
        return nn.Hardtanh(0.0, ACT_RAIL[act])
    if act == "tanh":
        return nn.Tanh()
    raise ValueError(f"未知激活 {act!r}，可选 {ACTIVATIONS}")


def _block(in_dim: int, out_dim: int, act: str = "relu") -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.BatchNorm1d(out_dim),
        _activation(act),
    )


class DenseAutoEncoder(nn.Module):
    """``center=True`` 时在 D/A 之前按维度减去一个固定偏置（训练集均值），输出后加回。

    log-mel 的直流约 -26 dB，起伏只有约 3.5 dB（std）。不去直流时，首层 D/A 与末层 A/D
    的 max-abs 量程大半花在直流上（每 batch 的 max|x| 约 59，去直流后约 20，约 1.5 bit）。
    去直流是重参数化：输入和重构目标减去同一个常数，损失与异常分数不变，
    变的只是转换器看到的信号范围。物理上对应 D/A 前的逐通道偏置（数字或模拟）。
    偏置由 :func:`src.train.train` 在训练集上算出，随 state_dict 保存。
    """

    def __init__(
        self,
        input_dim: int = 640,
        hidden: int = 128,
        n_blocks: int = 4,
        bottleneck: int = 8,
        center: bool = False,
        act: str = "relu",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.center = center
        if center:
            self.register_buffer("input_offset", torch.zeros(input_dim))
        if act not in ACTIVATIONS:
            raise ValueError(f"未知激活 {act!r}，可选 {ACTIVATIONS}")
        self.act = act
        self.act_rail = ACT_RAIL[act]    # 隐藏层输入范围；None = 不饱和（见 analog.apply_signal_ranges）

        enc: list[nn.Module] = []
        d = input_dim
        for _ in range(n_blocks):
            enc.append(_block(d, hidden, act))
            d = hidden
        enc.append(_block(d, bottleneck, act))
        self.encoder = nn.Sequential(*enc)

        dec: list[nn.Module] = []
        d = bottleneck
        for _ in range(n_blocks):
            dec.append(_block(d, hidden, act))
            d = hidden
        self.decoder = nn.Sequential(*dec)

        self.out = nn.Linear(d, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.center:
            return self.out(self.decoder(self.encoder(x)))
        return self.out(self.decoder(self.encoder(x - self.input_offset))) + self.input_offset

    @torch.no_grad()
    def set_input_offset(self, offset: torch.Tensor) -> None:
        if not self.center:
            raise RuntimeError("模型未启用 center，不能设置输入偏置")
        self.input_offset.copy_(offset.reshape(-1))


@torch.no_grad()
def reconstruction_error(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """逐样本重构 MSE，返回 shape ``[B]``。异常分数由它在文件内取均值得到。"""
    return ((model(x) - x) ** 2).mean(dim=1)


def build_from_config(cfg, input_dim: int) -> DenseAutoEncoder:
    return DenseAutoEncoder(
        input_dim=input_dim,
        hidden=int(cfg.model.hidden),
        n_blocks=int(cfg.model.n_blocks),
        bottleneck=int(cfg.model.bottleneck),
        center=bool(cfg.feature.get("center", False)),
        act=str(cfg.model.get("act", "relu")),
    )
