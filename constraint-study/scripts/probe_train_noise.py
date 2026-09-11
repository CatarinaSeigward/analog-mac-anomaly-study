"""Day 4 诊断：none 架构的器件噪声敏感性，由训练时还是部署时的激活噪声决定？

背景
    none 架构在 σ_read = 0 时 AUC 0.650、跨芯片 std 很大；σ_read = 4% 时 0.729 且稳定。
    已排除：评估 batch 组成（关噪声时最大差 0.0008）；逐片 BN 重标定只挽回约 0.02
    （scripts/probe_bn_calib.py）。

交叉评估
    用 σ_read = 0 与 σ_read = 4% 训练出的 none 模型，各自在 σ_read = 0 / 4% 下评估：
      - "4% 训练" 在 σ_read = 0 下仍然好 → 关键是**训练时**的激活噪声 —— 一条训练配方
      - "0 训练" 在 σ_read = 4% 下变好   → 关键是**部署时**的噪声 —— 与训练无关

    python scripts/probe_train_noise.py

产出
    results/train_noise_cross.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src.evaluate_analog import evaluate_chips  # noqa: E402
from src.experiment import TARGET, load_model, target_cfg  # noqa: E402
from src.features.logmel import cache_prefix, load_cache  # noqa: E402
from src.noise import DeviceParams  # noqa: E402
from src.utils import get_device, load_config  # noqa: E402

TRAIN_SIGMA_READS = (0.0, 0.04)
EVAL_SIGMA_READS = (0.0, 0.04)
SEEDS = (0, 1, 2)
N_CHIPS = 10


def main() -> int:
    base = load_config("configs/fast.yaml")
    device = get_device()
    results_dir = Path(base.output.results_dir)
    mel, meta = load_cache(cache_prefix(base.data.cache_dir, TARGET["n_mels"], "test"))

    rows = []
    for train_sr in TRAIN_SIGMA_READS:
        for seed in SEEDS:
            name = f"arch_none_r{train_sr:g}_d2_s{seed}"
            for eval_sr in EVAL_SIGMA_READS:
                p = DeviceParams(sigma_prog=0.03, sigma_d2d=0.02,
                                 sigma_read=eval_sr, adc_mode="none")
                model = load_model(target_cfg(base, name, seed), results_dir, name, device, p)
                df = evaluate_chips(model, mel, meta, TARGET["frames"], device, p,
                                    n_chips=N_CHIPS, machine_ids=[1, 2, 3, 4], max_fpr=0.1)
                rows += [{"train_sigma_read": train_sr, "eval_sigma_read": eval_sr,
                          "seed": seed, **r} for r in df.to_dict("records")]

    runs = pd.DataFrame(rows)
    runs.to_csv(results_dir / "train_noise_cross.csv", index=False)
    agg = (runs.groupby(["train_sigma_read", "eval_sigma_read"])
               .auc.agg(["mean", "std", "min"]).round(4))
    print("none 架构，深度 2，σ_prog 3% / σ_d2d 2%，3 种子 × 10 芯片")
    print(agg.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
