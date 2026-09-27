"""DCASE 2020 Task 2 (ToyCar) 文件发现与标签解析。

文件名约定：``{normal|anomaly}_id_{ID}_{index}.wav``
- machine id 用正则从 ``id_(\\d+)`` 抽取
- 标签由文件名是否含 ``anomaly`` 决定（train 划分全部为 normal）
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_ID_RE = re.compile(r"id_(\d+)")


@dataclass(frozen=True)
class AudioFile:
    path: Path
    machine_id: int
    label: int  # 0 = normal, 1 = anomaly


def _parse(path: Path) -> AudioFile:
    m = _ID_RE.search(path.name)
    if m is None:
        raise ValueError(
            f"文件名里找不到 machine id：{path.name}\n"
            f"期望形如 normal_id_01_00000000.wav，请检查解压后的目录是否正确。"
        )
    label = 1 if "anomaly" in path.name.lower() else 0
    return AudioFile(path=path, machine_id=int(m.group(1)), label=label)


def list_files(root: str | Path, subdirs: list[str]) -> list[AudioFile]:
    """扫描 ``root/subdir/*.wav``，返回按路径排序的文件列表。"""
    root = Path(root)
    files: list[AudioFile] = []
    missing: list[str] = []

    for sub in subdirs:
        d = root / sub
        if not d.is_dir():
            missing.append(str(d))
            continue
        wavs = sorted(d.glob("*.wav"))
        if not wavs:
            missing.append(f"{d}（目录存在但没有 .wav）")
        files.extend(_parse(p) for p in wavs)

    if missing:
        raise FileNotFoundError(
            "以下数据目录缺失或为空：\n  "
            + "\n  ".join(missing)
            + "\n\n请先按 根目录 README.md 的 Data 一节下载并解压数据集。"
        )
    return files


def summarize(files: list[AudioFile]) -> dict[int, dict[str, int]]:
    """按 machine id 统计 normal / anomaly 数量，用于核对下载完整性。"""
    out: dict[int, dict[str, int]] = {}
    for f in files:
        bucket = out.setdefault(f.machine_id, {"normal": 0, "anomaly": 0})
        bucket["anomaly" if f.label else "normal"] += 1
    return dict(sorted(out.items()))
