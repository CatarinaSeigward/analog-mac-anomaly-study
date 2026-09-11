"""在 N 颗仿真芯片上评估模拟模型。

★ 为什么必须跨多颗芯片评估（ROADMAP §2.1）
  器件间失配（D2D）是每颗芯片固有的固定偏差。只在一颗"平均芯片"上评估会
  严重高估性能，而且完全无法回答"量产良率如何"—— 而那是芯片公司看结果时
  问的第一个问题。

★ D2D 是评估期变量，不是训练期变量
  部署芯片的 D2D 在训练时未知，所以训练一次即可，然后部署到 N 颗仿真芯片上
  分别评估。不要为每颗芯片重新训练（那会把算力从 48 分钟变成 8 小时）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .evaluate import file_scores, per_machine_metrics
from .models.analog import set_chip, set_params
from .noise import DeviceParams


def evaluate_chips(
    model: torch.nn.Module,
    mel: np.ndarray,
    meta: dict,
    frames: int,
    device: torch.device,
    params: DeviceParams,
    n_chips: int = 10,
    machine_ids: list[int] | None = None,
    max_fpr: float = 0.1,
    trials: int = 1,
) -> pd.DataFrame:
    """逐芯片评估，返回每 (chip, trial) 一行的 DataFrame。

    ``trials`` > 1 时对同一颗芯片重复评估，用来分离 C2C 噪声的贡献
    （D2D 固定、C2C 每次前向重采样）。默认 1 —— 异常分数本身已经是
    文件内约 340 个窗口的均值，C2C 已被大量平均掉。
    """
    machine_ids = machine_ids or [1, 2, 3, 4]
    set_params(model, params)
    model.eval()

    rows = []
    for chip in range(n_chips):
        set_chip(model, chip)
        for t in range(trials):
            s, y, m = file_scores(model, mel, meta, frames, device)
            df = per_machine_metrics(s, y, m, machine_ids, max_fpr)
            mean_row = df.iloc[-1]
            row = {"chip": chip, "trial": t,
                   "auc": float(mean_row["auc"]), "pauc": float(mean_row["pauc"])}
            for _, r in df[df.machine_id != "mean"].iterrows():
                row[f"auc_id{int(r.machine_id)}"] = float(r.auc)
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_chips(df: pd.DataFrame) -> dict:
    """跨芯片聚合。★ 报 std 是必需的 —— 那是量产视角下的良率信息。"""
    return {
        "n_chips": int(df.chip.nunique()),
        "n_evals": int(len(df)),
        "auc_mean": float(df.auc.mean()),
        "auc_std": float(df.auc.std(ddof=1)) if len(df) > 1 else 0.0,
        "auc_min": float(df.auc.min()),
        "auc_max": float(df.auc.max()),
        "pauc_mean": float(df.pauc.mean()),
        "pauc_std": float(df.pauc.std(ddof=1)) if len(df) > 1 else 0.0,
    }
