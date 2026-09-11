"""Day 4 起的实验脚本共用工具：目标配置、checkpoint 重建、收敛性检查、聚合、分片。

``scripts/sweep_noise.py``（Day 3）保留了其中几个函数的早期副本，
以保证 Day 3 能原样复现；新脚本一律从这里导入。
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import pandas as pd
import torch
from omegaconf import OmegaConf

from .models.analog import analogize
from .models.autoencoder import build_from_config

# Day 2 选定的目标配置：6 mel 带 × 5 帧 = 30 维，hidden=30，2 个隐藏块，6 个 tile
TARGET = {"n_mels": 6, "frames": 5, "hidden": 30, "n_blocks": 2, "bottleneck": 4}


def target_cfg(base, name: str, seed: int, **model_overrides):
    """目标配置，可覆盖 model.* 字段（如 ``n_blocks=4`` 做深度扫描）。"""
    model = {k: TARGET[k] for k in ("hidden", "n_blocks", "bottleneck")}
    unknown = set(model_overrides) - set(model)
    if unknown:
        raise KeyError(f"未知的 model 覆盖项: {sorted(unknown)}")
    model.update(model_overrides)
    return OmegaConf.merge(base, OmegaConf.from_dotlist([
        f"name={name}", f"seed={seed}",
        f"feature.n_mels={TARGET['n_mels']}", f"feature.frames={TARGET['frames']}",
        *[f"model.{k}={v}" for k, v in model.items()],
    ]))


def checkpoint_exists(results_dir: Path, name: str) -> bool:
    return (Path(results_dir) / name / "model.pt").exists()


def load_model(cfg, results_dir: Path, name: str, device, params=None):
    """从 checkpoint 重建模型；``params`` 非 None 时换成 AnalogLinear 并放置 A/D。"""
    ckpt = torch.load(Path(results_dir) / name / "model.pt", map_location=device,
                      weights_only=False)
    model = build_from_config(cfg, input_dim=ckpt["input_dim"])
    if params is not None:
        analogize(model, params)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


# ---------------------------------------------------------------------------
# 收敛性检查
# ---------------------------------------------------------------------------
def final_val_loss(results_dir: Path, name: str) -> float | None:
    f = Path(results_dir) / name / "history.json"
    if not f.exists():
        return None
    return json.loads(f.read_text(encoding="utf-8"))["history"][-1]["val_loss"]


def find_nonconverged(results_dir: Path, groups: dict, factor: float = 2.0) -> dict[str, float]:
    """按**验证损失**识别未收敛的训练，**逐组**比较。

    ``groups``：{条件: [同一条件下不同种子的 run 名]}。
    返回 {run 名: 相对组中位数的倍数}。

    ★ 为什么用 val loss 而不是测试 AUC
      val loss 是训练期诊断量，剔除它不涉及测试集；按测试 AUC 剔除就是挑数据
      （PLAN 第 9 节诚信规范）。

    ★ 为什么逐组而不是全体中位数
      噪声越大 val loss 本来就越高（Day 3 里 σ=10% 的正常 run 约为全体中位数的
      1.6×）。σ_read 扫得更高时，全体中位数会把"条件本身难"误判成"没收敛"，
      也可能反过来掩盖低噪声组里的塌陷。

    少于 3 个 run 的组不做判定：只有 2 个时中位数是均值，一个塌陷会把中位数
    一起拉高，判据失效。
    """
    bad: dict[str, float] = {}
    for names in groups.values():
        losses = {n: v for n in names if (v := final_val_loss(results_dir, n)) is not None}
        if len(losses) < 3:
            continue
        med = statistics.median(losses.values())
        bad.update({n: v / med for n, v in losses.items() if v > factor * med})
    return bad


def report_nonconverged(bad: dict[str, float], factor: float, n_total: int) -> None:
    print()
    if not bad:
        print(f"[收敛性检查] {n_total} 次训练均收敛"
              f"（判据：final val loss > {factor}x 同组中位数；不足 3 个种子的组不判定）")
        return
    print(f"[收敛性检查] {len(bad)}/{n_total} 次训练未收敛，"
          f"判据 final val loss > {factor}x 同组中位数：")
    for n, r in sorted(bad.items(), key=lambda x: -x[1]):
        print(f"    {n:<36} {r:.2f}x  -> 从聚合中剔除")
    print("    （按验证损失剔除，不涉及测试集）")


# ---------------------------------------------------------------------------
# 聚合与分片
# ---------------------------------------------------------------------------
def aggregate(runs: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """按 ``keys`` 聚合跨 (种子 × 芯片) 的 mean ± std，附最差芯片。"""
    agg = (runs.groupby(keys)
               .agg(n=("auc", "size"),
                    n_seeds=("seed", "nunique"), n_chips=("chip", "nunique"),
                    auc_mean=("auc", "mean"), auc_std=("auc", "std"),
                    auc_min=("auc", "min"),
                    pauc_mean=("pauc", "mean"), pauc_std=("pauc", "std"))
               .reset_index())
    agg[["auc_std", "pauc_std"]] = agg[["auc_std", "pauc_std"]].fillna(0.0)
    return agg


def print_agg(agg: pd.DataFrame, keys: list[str]) -> None:
    t = agg.copy()
    t["AUC"] = t.apply(lambda r: f"{r.auc_mean:.4f} ± {r.auc_std:.4f}", axis=1)
    t["pAUC"] = t.apply(lambda r: f"{r.pauc_mean:.4f} ± {r.pauc_std:.4f}", axis=1)
    print(t[[*keys, "n_seeds", "n", "AUC", "pAUC", "auc_min"]].to_string(index=False))


def shard(jobs: list, spec: str | None) -> list:
    """``spec="K/N"`` 时只返回第 K 份（0 起、轮询分配），用于多进程并行训练。

    调用方应先按耗时降序排好 ``jobs``，轮询分配才会均衡。
    """
    if not spec:
        return list(jobs)
    k, n = (int(v) for v in spec.split("/"))
    if not 0 <= k < n:
        raise ValueError(f"分片 {spec!r} 不合法，应满足 0 <= K < N")
    return list(jobs)[k::n]
