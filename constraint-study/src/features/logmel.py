"""log-mel 特征提取与缓存。

★ 本模块必须与 MLPerf Tiny / DCASE2020 参考实现逐项对齐，否则 Day 1 复现不出
官方 AUC。参考实现的关键两行是：

    mel = librosa.feature.melspectrogram(y, sr, n_fft, hop_length, n_mels, power)
    log_mel = 20.0 / power * np.log10(mel + sys.float_info.epsilon)

窗口拼接约定（frame-major）：第 i 个窗口 = concat(mel[i], mel[i+1], ..., mel[i+F-1])，
即长度 n_mels * frames 的向量。把 log-mel 存成 [n_frames, n_mels] 的 C 连续数组后，
这个拼接恰好是内存中一段连续的 640 个 float —— 所以可以用 as_strided 零拷贝取窗口。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ..data.dcase import AudioFile


# --------------------------------------------------------------------------
# 单文件特征
# --------------------------------------------------------------------------
def compute_logmel(
    path: str | Path,
    n_mels: int = 128,
    n_fft: int = 1024,
    hop_length: int = 512,
    power: float = 2.0,
    backend: str = "librosa",
) -> np.ndarray:
    """返回 ``[n_frames, n_mels]`` 的 float32 C 连续数组。"""
    if backend == "librosa":
        import librosa

        y, sr = librosa.load(str(path), sr=None, mono=True)
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=power
        )
    elif backend == "torchaudio":
        # 近似匹配：torchaudio 默认 htk mel + 无归一化，与 librosa 默认不同，
        # 这里显式设成 slaney 以尽量对齐。仍可能有细微差异，只在 librosa 装不上时用。
        import torch
        import torchaudio

        wav, sr = torchaudio.load(str(path))
        wav = wav.mean(0, keepdim=True)
        mel_fn = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=power,
            norm="slaney",
            mel_scale="slaney",
        )
        mel = mel_fn(wav).squeeze(0).numpy()
    else:
        raise ValueError(f"未知 backend: {backend}")

    # ★ 与参考实现一致的 dB 变换。power=2.0 时等价于 10*log10(mel)
    log_mel = (20.0 / power) * np.log10(mel + sys.float_info.epsilon)

    # 转成 [n_frames, n_mels] 并保证 C 连续 —— make_windows 依赖这一点
    return np.ascontiguousarray(log_mel.T, dtype=np.float32)


# --------------------------------------------------------------------------
# 窗口（零拷贝）
# --------------------------------------------------------------------------
def make_windows(logmel_t: np.ndarray, frames: int) -> np.ndarray:
    """把 ``[n_frames, n_mels]`` 展开成 ``[n_windows, n_mels*frames]`` 的**视图**。

    注意返回的是 as_strided 视图，与输入共享内存；不要原地修改。
    """
    if logmel_t.ndim != 2:
        raise ValueError(f"期望二维 [n_frames, n_mels]，得到 {logmel_t.shape}")
    if not logmel_t.flags["C_CONTIGUOUS"]:
        raise ValueError("logmel_t 必须是 C 连续的，否则 as_strided 的步长假设不成立")

    n_frames, n_mels = logmel_t.shape
    n_win = n_frames - frames + 1
    if n_win <= 0:
        return np.empty((0, n_mels * frames), dtype=logmel_t.dtype)

    row_stride, item_stride = logmel_t.strides
    return np.lib.stride_tricks.as_strided(
        logmel_t,
        shape=(n_win, n_mels * frames),
        strides=(row_stride, item_stride),
        writeable=False,
    )


# --------------------------------------------------------------------------
# 缓存
# --------------------------------------------------------------------------
def build_cache(
    files: list[AudioFile],
    out_prefix: str | Path,
    n_mels: int = 128,
    n_fft: int = 1024,
    hop_length: int = 512,
    power: float = 2.0,
    backend: str = "librosa",
    desc: str = "extracting",
) -> tuple[np.ndarray, dict]:
    """把所有文件的 log-mel 拼成一个大数组存盘，附带偏移索引。

    产出：
      ``{prefix}_logmel.npy``  float32 [total_frames, n_mels]
      ``{prefix}_meta.json``   每个文件的 offset / n_frames / machine_id / label
    """
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    chunks: list[np.ndarray] = []
    entries: list[dict] = []
    offset = 0

    for f in tqdm(files, desc=desc, unit="file"):
        lm = compute_logmel(f.path, n_mels, n_fft, hop_length, power, backend)
        chunks.append(lm)
        entries.append(
            {
                "path": str(f.path),
                "offset": offset,
                "n_frames": int(lm.shape[0]),
                "machine_id": f.machine_id,
                "label": f.label,
            }
        )
        offset += lm.shape[0]

    data = np.concatenate(chunks, axis=0) if chunks else np.empty((0, n_mels), np.float32)
    meta = {
        "n_mels": n_mels,
        "n_fft": n_fft,
        "hop_length": hop_length,
        "power": power,
        "backend": backend,
        "total_frames": int(data.shape[0]),
        "files": entries,
    }

    np.save(f"{out_prefix}_logmel.npy", data)
    with open(f"{out_prefix}_meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    return data, meta


def load_cache(prefix: str | Path, mmap: bool = False) -> tuple[np.ndarray, dict]:
    prefix = Path(prefix)
    npy = Path(f"{prefix}_logmel.npy")
    js = Path(f"{prefix}_meta.json")
    if not npy.exists() or not js.exists():
        raise FileNotFoundError(
            f"缓存不存在：{npy} / {js}\n请先运行 scripts/prepare_data.py"
        )
    data = np.load(npy, mmap_mode="r" if mmap else None)
    with open(js, encoding="utf-8") as fh:
        meta = json.load(fh)
    return data, meta


def window_start_rows(meta: dict, frames: int, stride: int = 1) -> np.ndarray:
    """所有文件的合法窗口起始行号（在拼接后的大数组里的绝对行号）。

    窗口不跨文件边界。返回 int64 数组。

    ``stride`` 用于训练时抽稀：相邻窗口重叠 ``frames-1`` 帧，信息高度冗余，
    取每第 k 个窗口可以在几乎不损失信息的前提下把训练集缩小 k 倍。
    ★ 只用于训练。评估必须 stride=1（异常分数定义为文件内**所有**窗口的均值）。
    """
    if stride < 1:
        raise ValueError(f"stride 必须 >= 1，得到 {stride}")
    starts: list[np.ndarray] = []
    for e in meta["files"]:
        n_win = e["n_frames"] - frames + 1
        if n_win > 0:
            starts.append(
                np.arange(e["offset"], e["offset"] + n_win, stride, dtype=np.int64)
            )
    return np.concatenate(starts) if starts else np.empty(0, dtype=np.int64)


def cache_prefix(cache_dir: str | Path, n_mels: int, split: str) -> Path:
    """缓存路径按 n_mels 分目录。

    mel 滤波器组不能从 128 带降采样得到，换 n_mels 必须重新提取特征，
    所以每个 n_mels 值有自己的一份缓存。frames 不影响缓存（只是窗口宽度）。
    """
    return Path(cache_dir) / f"n_mels{n_mels}" / split
