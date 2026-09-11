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


def _block(in_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.BatchNorm1d(out_dim),
        nn.ReLU(inplace=True),
    )


class DenseAutoEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 640,
        hidden: int = 128,
        n_blocks: int = 4,
        bottleneck: int = 8,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim

        enc: list[nn.Module] = []
        d = input_dim
        for _ in range(n_blocks):
            enc.append(_block(d, hidden))
            d = hidden
        enc.append(_block(d, bottleneck))
        self.encoder = nn.Sequential(*enc)

        dec: list[nn.Module] = []
        d = bottleneck
        for _ in range(n_blocks):
            dec.append(_block(d, hidden))
            d = hidden
        self.decoder = nn.Sequential(*dec)

        self.out = nn.Linear(d, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.decoder(self.encoder(x)))


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
    )
