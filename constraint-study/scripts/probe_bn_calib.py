"""Day 4 诊断：逐片 BN 重标定能否消除 none / σ_read=0 的器件失配敏感性。

背景
    图 3 中 none 架构在 σ_read = 0 时 AUC 0.650 ± 0.058，σ_read = 4% 时反而 0.729 ± 0.013。
    - 已排除评估 batch 组成的影响：噪声全关时三种 batch 方式最大差仅 0.0008。
    - 该模型关噪声评估为 0.676，带噪声（10 颗芯片）降到 0.597（seed 0）——
      它对器件噪声特别敏感，而同条件的 per_layer 模型和 σ_read=4% 的 none 模型都不敏感。

假设
    训练时 BN 用当前 batch 的统计量，等于在每个 batch 内"顺手吸收"了当次 D2D
    采样造成的激活偏移；部署时 BN 改用固定的 running stats，吸收不了某颗芯片
    **固定**的偏移。

检验
    每颗芯片用少量**正常**数据重估 BN 统计量（= assumptions A5 的逐片标定；
    部署时只需采集设备正常运转的声音，不需要任何异常样本），再评估。
    若 none / σ_read=0 恢复到与其他条件相当的水平，假设成立。

    python scripts/probe_bn_calib.py
    python scripts/probe_bn_calib.py --calib-windows 2048 20480

产出
    results/bn_calib_runs.csv / results/bn_calib.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.evaluate import file_scores, per_machine_metrics  # noqa: E402
from src.experiment import TARGET, load_model, target_cfg  # noqa: E402
from src.features.logmel import cache_prefix, load_cache, window_start_rows  # noqa: E402
from src.models.analog import set_chip, set_params  # noqa: E402
from src.noise import DeviceParams  # noqa: E402
from src.utils import get_device, load_config  # noqa: E402

CASES = [("none", 0.0), ("per_layer", 0.0), ("none", 0.04)]
SEEDS = [0, 1, 2]
N_CHIPS = 10
SIGMA_PROG, SIGMA_D2D = 0.03, 0.02
FRAMES = TARGET["frames"]


def calib_windows(base, device, n: int, seed: int = 0) -> torch.Tensor:
    """从**训练集（全部正常样本）**里随机取 n 个窗口，作为标定数据。"""
    mel, meta = load_cache(cache_prefix(base.data.cache_dir, TARGET["n_mels"], "train"))
    starts = window_start_rows(meta, FRAMES)
    pick = np.random.default_rng(seed).choice(len(starts), n, replace=False)
    rows = starts[pick][:, None] + np.arange(FRAMES)[None, :]
    x = np.asarray(mel)[rows].reshape(n, -1)
    return torch.as_tensor(x, dtype=torch.float32, device=device)


class BNState:
    """保存 / 恢复 / 重估模型里全部 BatchNorm 的 running 统计量。"""

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.bns = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm1d)]
        self.saved = [(b.running_mean.clone(), b.running_var.clone(),
                       b.num_batches_tracked.clone(), b.momentum) for b in self.bns]

    def restore(self) -> None:
        for b, (rm, rv, nb, mom) in zip(self.bns, self.saved):
            b.running_mean.copy_(rm)
            b.running_var.copy_(rv)
            b.num_batches_tracked.copy_(nb)
            b.momentum = mom

    @torch.no_grad()
    def recalibrate(self, x: torch.Tensor, bs: int = 2048) -> None:
        """在当前芯片（D2D 已固定）上重估 BN 统计量。只动 BN，不动权重。"""
        for b in self.bns:
            b.reset_running_stats()
            b.momentum = None               # 累积平均，与样本顺序无关
        self.model.train()                  # 只影响 BN；AnalogLinear 的行为由 chip_id 决定
        for i in range(0, len(x), bs):
            self.model(x[i:i + bs])
        self.model.eval()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/fast.yaml")
    ap.add_argument("--calib-windows", type=int, nargs="+", default=[2048, 20480],
                    help="每颗芯片标定用的正常窗口数。2048 ≈ 6 个 10 秒文件 ≈ 1 分钟音频")
    ap.add_argument("--n-chips", type=int, default=N_CHIPS)
    args = ap.parse_args()

    base = load_config(args.config)
    device = get_device()
    results_dir = Path(base.output.results_dir)
    mel, meta = load_cache(cache_prefix(base.data.cache_dir, TARGET["n_mels"], "test"))
    calib = {n: calib_windows(base, device, n) for n in args.calib_windows}

    rows = []
    t0 = time.time()
    for arch, sr in CASES:
        p = DeviceParams(sigma_prog=SIGMA_PROG, sigma_d2d=SIGMA_D2D,
                         sigma_read=sr, adc_mode=arch)
        for seed in SEEDS:
            name = f"arch_{arch}_r{sr:g}_d2_s{seed}"
            model = load_model(target_cfg(base, name, seed), results_dir, name, device, p)
            set_params(model, p)
            bn = BNState(model)
            for chip in range(args.n_chips):
                set_chip(model, chip)
                for n_cal in [0, *args.calib_windows]:
                    bn.restore()
                    if n_cal:
                        bn.recalibrate(calib[n_cal])
                    torch.manual_seed(1000 * seed + chip)   # 各条件共用同一组 C2C 噪声种子
                    s, y, m = file_scores(model, mel, meta, FRAMES, device)
                    auc = float(per_machine_metrics(s, y, m, [1, 2, 3, 4], 0.1).iloc[-1]["auc"])
                    rows.append({"arch": arch, "sigma_read": sr, "seed": seed,
                                 "chip": chip, "calib_windows": n_cal, "auc": auc})
            bn.restore()
        print(f"  {arch:<10} σ_read={sr:<5g} 完成 ({time.time() - t0:.0f}s)")

    runs = pd.DataFrame(rows)
    runs.to_csv(results_dir / "bn_calib_runs.csv", index=False)
    agg = (runs.groupby(["arch", "sigma_read", "calib_windows"])
               .agg(n=("auc", "size"), auc_mean=("auc", "mean"),
                    auc_std=("auc", "std"), auc_min=("auc", "min"))
               .reset_index())
    agg.to_csv(results_dir / "bn_calib.csv", index=False)

    print("\n逐片 BN 重标定（calib_windows = 0 表示不标定，沿用训练时的 running stats）")
    t = agg.copy()
    t["AUC"] = t.apply(lambda r: f"{r.auc_mean:.4f} ± {r.auc_std:.4f}", axis=1)
    print(t[["arch", "sigma_read", "calib_windows", "n", "AUC", "auc_min"]].to_string(index=False))

    print("\n同一 (种子, 芯片) 配对：标定后 − 标定前")
    base_auc = runs[runs.calib_windows == 0].set_index(["arch", "sigma_read", "seed", "chip"]).auc
    for n_cal in args.calib_windows:
        cal = runs[runs.calib_windows == n_cal].set_index(["arch", "sigma_read", "seed", "chip"]).auc
        d = (cal - base_auc).groupby(level=["arch", "sigma_read"]).agg(["mean", "std"])
        for (arch, sr), r in d.iterrows():
            print(f"  {n_cal:>6} 窗口  {arch:<10} σ_read={sr:<5g} {r['mean']:+.4f} ± {r['std']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
