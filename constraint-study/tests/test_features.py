"""特征模块单元测试。

重点是 make_windows 的 as_strided 实现 —— 这是最容易写错、且错了之后
所有下游结果都无声地变错的地方。
"""

from __future__ import annotations

import numpy as np
import pytest

from src.features.logmel import make_windows, window_start_rows


def _naive_windows(logmel_t: np.ndarray, frames: int) -> np.ndarray:
    """DCASE 参考实现的朴素写法，作为 ground truth。

    参考代码里 log_mel 是 [n_mels, n_frames]，这里的 logmel_t 是它的转置。
        vectorarray[:, n_mels*t : n_mels*(t+1)] = log_mel[:, t : t+size].T
    """
    n_frames, n_mels = logmel_t.shape
    size = n_frames - frames + 1
    out = np.zeros((size, n_mels * frames), dtype=logmel_t.dtype)
    for t in range(frames):
        out[:, n_mels * t : n_mels * (t + 1)] = logmel_t[t : t + size, :]
    return out


@pytest.mark.parametrize("n_frames,n_mels,frames", [(20, 8, 5), (313, 128, 5), (10, 4, 1), (7, 3, 7)])
def test_make_windows_matches_naive(n_frames, n_mels, frames):
    rng = np.random.default_rng(0)
    lm = np.ascontiguousarray(rng.standard_normal((n_frames, n_mels)), dtype=np.float32)
    np.testing.assert_allclose(make_windows(lm, frames), _naive_windows(lm, frames), rtol=0, atol=0)


def test_make_windows_shape_and_content():
    lm = np.arange(6 * 3, dtype=np.float32).reshape(6, 3)  # [n_frames=6, n_mels=3]
    w = make_windows(lm, frames=2)
    assert w.shape == (5, 6)
    # 第 0 个窗口 = 第 0、1 帧拼接
    np.testing.assert_array_equal(w[0], np.array([0, 1, 2, 3, 4, 5], dtype=np.float32))
    # 第 3 个窗口 = 第 3、4 帧拼接
    np.testing.assert_array_equal(w[3], np.array([9, 10, 11, 12, 13, 14], dtype=np.float32))


def test_make_windows_too_short_returns_empty():
    lm = np.zeros((3, 4), dtype=np.float32)
    assert make_windows(lm, frames=5).shape == (0, 20)


def test_make_windows_rejects_non_contiguous():
    lm = np.zeros((10, 8), dtype=np.float32).T  # 转置后非 C 连续
    with pytest.raises(ValueError, match="C 连续"):
        make_windows(lm, frames=5)


def test_window_start_rows_does_not_cross_file_boundary():
    meta = {
        "files": [
            {"offset": 0, "n_frames": 10},
            {"offset": 10, "n_frames": 6},
            {"offset": 16, "n_frames": 3},  # < frames，应被跳过
        ]
    }
    starts = window_start_rows(meta, frames=5)
    # 文件0 贡献 6 个 (0..5)，文件1 贡献 2 个 (10,11)，文件2 贡献 0 个
    np.testing.assert_array_equal(starts, np.array([0, 1, 2, 3, 4, 5, 10, 11]))
    # 关键：不能出现跨文件边界的起始行（6..9 会让窗口越过文件0的末尾）
    assert not set(range(6, 10)) & set(starts.tolist())


# ---------------------------------------------------------------------------
# 窗口抽稀（Day 2 fast 配置用）
# ---------------------------------------------------------------------------
def test_window_start_rows_stride():
    """stride=k 应取每第 k 个窗口，且仍不跨文件边界。"""
    meta = {"files": [{"offset": 0, "n_frames": 10}, {"offset": 10, "n_frames": 8}]}
    # 文件0 合法起点 0..5，文件1 合法起点 10..13
    np.testing.assert_array_equal(
        window_start_rows(meta, frames=5, stride=1),
        np.array([0, 1, 2, 3, 4, 5, 10, 11, 12, 13]),
    )
    np.testing.assert_array_equal(
        window_start_rows(meta, frames=5, stride=2),
        np.array([0, 2, 4, 10, 12]),
    )
    np.testing.assert_array_equal(
        window_start_rows(meta, frames=5, stride=3),
        np.array([0, 3, 10, 13]),
    )


def test_window_start_rows_stride_never_crosses_boundary():
    meta = {"files": [{"offset": 0, "n_frames": 20}, {"offset": 20, "n_frames": 20}]}
    for stride in (1, 2, 3, 5, 7):
        starts = window_start_rows(meta, frames=5, stride=stride)
        # 文件0 的窗口起点不得超过 15（否则窗口越界到文件1）
        assert ((starts <= 15) | (starts >= 20)).all()
        assert (starts <= 35).all()


def test_window_start_rows_rejects_bad_stride():
    meta = {"files": [{"offset": 0, "n_frames": 10}]}
    with pytest.raises(ValueError, match="stride"):
        window_start_rows(meta, frames=5, stride=0)
