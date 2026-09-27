"""NEXT-STEPS 0-3：固定阈值下的逐片工作点 —— 阈值出厂定一次够不够，还是每台安装时要重定？

背景
    AUC 与阈值无关，它看不见芯片间的差异对**工作点**的影响。部署时阈值是一个具体数字，
    Ai Linear 对外的指标也是 precision（>92 %），这是工作点指标。报告 §4.1 的交付表里
    "Decision threshold" 一行一直没有数字。

设计
    每个 machine ID 的**测试集正常片段**按固定种子对半分：
      calib 一半  —— 定阈值（模拟安装时录到的正常运转声音，模型没见过）
      eval  一半  —— 算 FPR；异常片段全部用来算 TPR
    训练集片段不用来定阈值：模型见过它们，得分偏低，阈值会过紧。

    阈值 = calib 正常片段得分的 (1 - target_fpr) 分位数，逐 machine ID：
      factory  出厂阈值 —— 用无噪声模型定一次，所有芯片共用（衡量仿真到实物的差距）
      golden   参考芯片阈值 —— 在芯片 0 上定一次，用到芯片 1–9（衡量芯片间能否共用一个阈值）
      install  安装阈值 —— 每颗芯片用自己的得分各定一次（只需读最终得分，不需要隐藏层读出）
    另记录每颗芯片正常片段得分的中位数相对无噪声模型的倍数（噪声让得分整体上移多少）。
    每颗芯片带标称噪声（σ_prog 3 %、σ_d2d 2 %、σ_read 4 %），10 颗 × 3 seeds。
    precision 取决于异常先验：测试集异常占 43 %，现场低得多，所以主报 FPR / TPR，
    再换算给定先验下的 precision。

    模型：STE 修复后的 6-bit 芯片模型；D/A 前去直流的 4-bit 候选模型。

    python scripts/probe_operating_point.py

产出
    results/oppoint_runs.csv / results/oppoint.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from src.evaluate import file_scores  # noqa: E402
from src.experiment import TARGET, checkpoint_exists, load_model, target_cfg  # noqa: E402
from src.features.logmel import cache_prefix, load_cache  # noqa: E402
from src.models.analog import set_chip, set_params  # noqa: E402
from src.noise import DeviceParams  # noqa: E402
from src.utils import get_device, load_config  # noqa: E402

SEEDS = [0, 1, 2]
N_CHIPS = 10
MACHINE_IDS = [1, 2, 3, 4]
PREVALENCES = [0.01, 0.10]
SPLIT_SEED = 20260922
# (模型标签, run 名前缀, 位宽, 是否去直流)
MODELS = [
    ("6bit", "bits_wsb6_r0.04_stefix", 6, False),
    ("4bit_dc_removed", "bits_wsb4_r0.04_stefix_c", 4, True),
]


def split_normals(labels: np.ndarray, mids: np.ndarray) -> np.ndarray:
    """返回布尔数组：True = 用来定阈值的正常片段（每个 machine ID 的一半）。"""
    rng = np.random.default_rng(SPLIT_SEED)
    calib = np.zeros(len(labels), dtype=bool)
    for mid in MACHINE_IDS:
        idx = np.flatnonzero((mids == mid) & (labels == 0))
        calib[rng.choice(idx, len(idx) // 2, replace=False)] = True
    return calib


def thresholds(scores, labels, mids, calib, target_fpr) -> dict[int, float]:
    return {mid: float(np.quantile(scores[calib & (mids == mid)], 1 - target_fpr))
            for mid in MACHINE_IDS}


def rates(scores, labels, mids, calib, thr: dict[int, float]) -> dict:
    """逐 machine ID 的 FPR（eval 那一半正常片段）与 TPR（全部异常），再对 ID 取平均。"""
    fpr, tpr = [], []
    for mid in MACHINE_IDS:
        m = mids == mid
        neg = m & (labels == 0) & ~calib
        pos = m & (labels == 1)
        fpr.append(float((scores[neg] > thr[mid]).mean()))
        tpr.append(float((scores[pos] > thr[mid]).mean()))
    return {"fpr": float(np.mean(fpr)), "tpr": float(np.mean(tpr))}


def precision(tpr: float, fpr: float, p: float) -> float:
    den = tpr * p + fpr * (1 - p)
    return tpr * p / den if den > 0 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/fast.yaml")
    ap.add_argument("--target-fpr", type=float, default=0.10)
    ap.add_argument("--device", default=None, help="如 cpu；默认自动选择")
    args = ap.parse_args()

    base = load_config(args.config)
    device = torch.device(args.device) if args.device else get_device()
    results_dir = Path(base.output.results_dir)
    mel, meta = load_cache(cache_prefix(base.data.cache_dir, TARGET["n_mels"], "test"))
    frames = TARGET["frames"]

    rows = []
    for label, prefix, bits, center in MODELS:
        mbase = OmegaConf.merge(base, {"feature": {"center": True}}) if center else base
        for seed in SEEDS:
            name = f"{prefix}_s{seed}"
            if not checkpoint_exists(results_dir, name):
                print(f"[跳过] 缺 checkpoint：{name}")
                continue
            quiet = DeviceParams(sigma_prog=0.0, sigma_d2d=0.0, sigma_read=0.0, adc_mode="none",
                                 w_bits=bits, s_bits=bits, b_bits=bits)
            nominal = quiet.with_(sigma_prog=0.03, sigma_d2d=0.02, sigma_read=0.04)
            model = load_model(target_cfg(mbase, name, seed), results_dir, name, device, quiet)

            s_q, labels, mids = file_scores(model, mel, meta, frames, device)
            calib = split_normals(labels, mids)
            thr_factory = thresholds(s_q, labels, mids, calib, args.target_fpr)

            med_quiet = float(np.median(s_q[calib]))
            set_params(model, nominal)
            thr_golden = None
            for chip in range(N_CHIPS):
                set_chip(model, chip)
                s, _, _ = file_scores(model, mel, meta, frames, device)
                thr_install = thresholds(s, labels, mids, calib, args.target_fpr)
                modes = [("factory", thr_factory), ("install", thr_install)]
                if chip == 0:
                    thr_golden = thr_install          # 参考芯片：芯片 0 自己不参与 golden 的统计
                else:
                    modes.append(("golden", thr_golden))
                shift = float(np.median(s[calib])) / med_quiet
                for mode, thr in modes:
                    r = rates(s, labels, mids, calib, thr)
                    rows.append({"model": label, "seed": seed, "chip": chip, "mode": mode, **r,
                                 "score_shift": shift,
                                 **{f"precision@{p:g}": precision(r["tpr"], r["fpr"], p)
                                    for p in PREVALENCES}})
            print(f"[done] {name}")

    runs = pd.DataFrame(rows)
    runs.to_csv(results_dir / "oppoint_runs.csv", index=False)
    cols = ["fpr", "tpr", "score_shift", *[f"precision@{p:g}" for p in PREVALENCES]]
    agg = runs.groupby(["model", "mode"])[cols].agg(["mean", "min", "max"]).round(3)
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.reset_index()
    agg.to_csv(results_dir / "oppoint.csv", index=False)

    pd.set_option("display.width", 250)
    print(f"\n目标 FPR = {args.target_fpr:g}；阈值逐 machine ID；{N_CHIPS} 颗芯片 × {runs.seed.nunique()} seeds，"
          "min / max 是跨 (芯片 × seed) 的最差与最好")
    print(agg.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
