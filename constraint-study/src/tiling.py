"""30x30 模拟 MAC 阵列的分块（tiling）与预算计算。

一个权重矩阵 ``[out_features, in_features]`` 需要的 tile 数::

    tiles = ceil(out_features / T) * ceil(in_features / T)

两个方向的分块代价不同（见 PLAN 附录 G.4）：
  - **输出维度分块**：多个 tile 各算一部分输出，互不相干，几乎无代价
  - **输入维度分块**：多个 tile 算部分和，结果需要相加。模拟域里就是把电流接到
    一起（KCL），电路上很便宜，**但参与的器件变多了 -> 累积噪声更大**

所以输入维度方向的分块有真实权衡：更多输入维度 = 更多信息，也 = 更多噪声源。
"""

from __future__ import annotations

import math

TILE = 30  # Ai Linear NN IC 的权重矩阵尺寸（6-bit W/S/B）


def layer_tiles(out_features: int, in_features: int, tile: int = TILE) -> int:
    """单个全连接层占用的 tile 数。"""
    return math.ceil(out_features / tile) * math.ceil(in_features / tile)


def layer_shapes(
    input_dim: int, hidden: int, n_blocks: int, bottleneck: int
) -> list[tuple[int, int]]:
    """DenseAutoEncoder 的全部 (out_features, in_features)，与 models/autoencoder.py 对应。"""
    shapes: list[tuple[int, int]] = []
    d = input_dim
    for _ in range(n_blocks):          # encoder
        shapes.append((hidden, d)); d = hidden
    shapes.append((bottleneck, d)); d = bottleneck   # bottleneck
    for _ in range(n_blocks):          # decoder
        shapes.append((hidden, d)); d = hidden
    shapes.append((input_dim, d))      # output
    return shapes


def model_tiles(
    input_dim: int, hidden: int, n_blocks: int, bottleneck: int, tile: int = TILE
) -> int:
    """整个自编码器占用的 tile 总数。"""
    return sum(layer_tiles(o, i, tile) for o, i in layer_shapes(input_dim, hidden, n_blocks, bottleneck))


def max_layer_tiles(
    input_dim: int, hidden: int, n_blocks: int, bottleneck: int, tile: int = TILE
) -> int:
    """单层最大 tile 占用 —— 决定芯片一次能不能装下这一层。"""
    return max(layer_tiles(o, i, tile) for o, i in layer_shapes(input_dim, hidden, n_blocks, bottleneck))


def tiled_matmul(x, w, tile: int = TILE):
    """按 tile 分块做矩阵乘，用于验证分块实现与稠密实现数值等价。

    ``x``: [B, in_features]，``w``: [out_features, in_features]，返回 [B, out_features]。
    无噪声无量化时必须与 ``x @ w.T`` 完全一致（见 tests/test_tiling.py）。
    """
    import torch

    out_f, in_f = w.shape
    y = torch.zeros(x.shape[0], out_f, dtype=x.dtype, device=x.device)
    for o0 in range(0, out_f, tile):
        o1 = min(o0 + tile, out_f)
        acc = torch.zeros(x.shape[0], o1 - o0, dtype=x.dtype, device=x.device)
        for i0 in range(0, in_f, tile):          # 输入维度分块 -> 部分和相加（模拟域即电流求和）
            i1 = min(i0 + tile, in_f)
            acc = acc + x[:, i0:i1] @ w[o0:o1, i0:i1].T
        y[:, o0:o1] = acc
    return y


if __name__ == "__main__":
    print(f"tile = {TILE}x{TILE}\n")
    for name, cfg in [
        ("C0 官方基线", (640, 128, 4, 8)),
        ("目标压缩配置", (30, 30, 2, 4)),
    ]:
        tot = model_tiles(*cfg)
        mx = max_layer_tiles(*cfg)
        print(f"{name}  input_dim={cfg[0]} hidden={cfg[1]} blocks={cfg[2]} bottleneck={cfg[3]}")
        for o, i in layer_shapes(*cfg):
            print(f"    FC({i:>4} -> {o:>4})   {layer_tiles(o, i):>4} tiles")
        print(f"    {'合计':<20} {tot:>4} tiles   单层最大 {mx}\n")
