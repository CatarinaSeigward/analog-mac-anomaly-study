"""分块（tiling）测试。

★ 最关键的一条：无噪声、无量化时，分块实现必须与稠密矩阵乘法数值等价。
   这里最容易出的 bug 是转置方向、部分和累加顺序、边界 tile 的处理。
   这个测试挂了，后面所有带噪声的结果都是错的。
"""

from __future__ import annotations

import torch

from src.tiling import layer_tiles, max_layer_tiles, model_tiles, tiled_matmul


def test_tiled_matmul_equivalence():
    torch.manual_seed(0)
    for out_f, in_f in [(30, 30), (64, 90), (128, 640), (8, 128), (7, 13), (31, 61)]:
        w = torch.randn(out_f, in_f, dtype=torch.float64)
        x = torch.randn(8, in_f, dtype=torch.float64)
        torch.testing.assert_close(tiled_matmul(x, w, tile=30), x @ w.T)


def test_tiled_matmul_tile_size_invariant():
    """不同 tile 尺寸应给出相同结果（无噪声时分块只是计算顺序不同）。"""
    torch.manual_seed(1)
    w = torch.randn(70, 100, dtype=torch.float64)
    x = torch.randn(4, 100, dtype=torch.float64)
    ref = x @ w.T
    for tile in (7, 10, 30, 64, 128):
        torch.testing.assert_close(tiled_matmul(x, w, tile=tile), ref)


def test_layer_tiles():
    assert layer_tiles(30, 30) == 1
    assert layer_tiles(128, 640) == 5 * 22       # ceil(128/30) * ceil(640/30)
    assert layer_tiles(31, 31) == 4              # 边界：刚超一格 -> 2x2
    assert layer_tiles(8, 128) == 1 * 5


def test_model_tiles_baseline_vs_squeezed():
    """C0 基线 380 tiles vs 目标配置 6 tiles —— app note 的核心对比。"""
    assert model_tiles(640, 128, 4, 8) == 380
    assert max_layer_tiles(640, 128, 4, 8) == 110
    assert model_tiles(30, 30, 2, 4) == 6
    assert max_layer_tiles(30, 30, 2, 4) == 1
