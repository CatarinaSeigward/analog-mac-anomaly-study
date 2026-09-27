"""NEXT-STEPS 0-2：训练时注入的噪声是不是一种正则化？全 fp32 对照（2026-09-22）。

背景
    1. 结论 5 / A12：用 4% 激活噪声训练的 none 模型，在安静芯片上比 fp32 高约 0.02（0.738 vs 0.720），
       一直没证实。它是噪声本身的正则化作用，还是训练-部署条件的匹配？
    2. 0-1：D/A 前去直流后，S = 4 bit 反而比 6 bit 高 0.036（0.762 vs 0.726）。不用模型的检测器
       解释不了（模板距离 0.58、纯量化残差 0.62）。猜测是去噪自编码器效应：粗量化的输入 +
       干净的重构目标，相当于训练时加了约 0.84 dB 的输入噪声（去直流后 4-bit LSB 2.89 dB / √12）。

设计
    全 fp32：不量化、没有器件噪声，只改训练时注入的噪声；全部在安静条件下（纯 fp32 前向）评估。
      fp32       基准
      act{σ}     激活噪声 σ_read = 2 / 4 / 8 %（与仿真器同一实现：std = σ · mean|W·x|，每层输出）
      in{std}    输入噪声 std = 0.42 / 0.84 / 1.68 dB，重构目标保持干净（train.input_noise）
    3 个 seed，同 seed 配对比较。
      - act 组高于 fp32              → A12 的 +0.02 来自正则化，与量化、芯片无关
      - in0.84 接近 0.76              → 0-1 的 S = 4 去直流之所以高于 6 bit，是去噪正则化
      - 两者都不高                    → 需要换解释

    python scripts/probe_noise_reg.py --shard 0/3      # 多进程训练
    python scripts/probe_noise_reg.py --eval-only      # 统一评估

产出
    results/noise_reg_runs.csv / results/noise_reg.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from src.evaluate_analog import evaluate_chips  # noqa: E402
from src.experiment import (  # noqa: E402
    TARGET, aggregate, checkpoint_exists, find_nonconverged, load_model,
    print_agg, report_nonconverged, shard, target_cfg,
)
from src.features.logmel import cache_prefix, load_cache  # noqa: E402
from src.noise import IDEAL, DeviceParams  # noqa: E402
from src.train import train  # noqa: E402
from src.utils import get_device, load_config  # noqa: E402

SEEDS = [0, 1, 2]
# (组名, 激活噪声 σ_read 或 None, 输入噪声 std)
ARMS = [
    ("fp32", None, 0.0),
    ("act0.02", 0.02, 0.0), ("act0.04", 0.04, 0.0), ("act0.08", 0.08, 0.0),
    ("in0.42", None, 0.42), ("in0.84", None, 0.84), ("in1.68", None, 1.68),
]


def run_name(arm: str, seed: int) -> str:
    return f"nreg_{arm}_s{seed}"


def act_params(sigma_read: float) -> DeviceParams:
    """全 fp32 + 激活噪声：不量化、无权重噪声，只有每层输出的读噪声。"""
    return DeviceParams(w_bits=0, s_bits=0, b_bits=0, sigma_prog=0.0, sigma_d2d=0.0,
                        sigma_read=sigma_read, adc_mode="none")


def job_cfg(base, j: dict):
    cfg = target_cfg(base, j["name"], j["seed"])
    if j["input_noise"] > 0:
        cfg = OmegaConf.merge(cfg, {"train": {"input_noise": j["input_noise"]}})
    return cfg


def evaluate_all(base, results_dir: Path, jobs: list[dict], factor: float) -> None:
    device = get_device()
    mel, meta = load_cache(cache_prefix(base.data.cache_dir, TARGET["n_mels"], "test"))
    rows, missing = [], []
    for j in jobs:
        if not checkpoint_exists(results_dir, j["name"]):
            missing.append(j["name"])
            continue
        # 安静条件：IDEAL 下 AnalogLinear 与 nn.Linear 数值一致，没有 D2D，一颗"芯片"即可
        model = load_model(job_cfg(base, j), results_dir, j["name"], device, IDEAL)
        df = evaluate_chips(model, mel, meta, TARGET["frames"], device, IDEAL, n_chips=1,
                            machine_ids=list(base.eval.machine_ids),
                            max_fpr=float(base.eval.max_fpr))
        rows += [{"arm": j["arm"], "seed": j["seed"], "run_name": j["name"], **r}
                 for r in df.to_dict("records")]
    if missing:
        print(f"[注意] {len(missing)} 个 checkpoint 缺失，未纳入：{', '.join(missing)}")
    if not rows:
        return

    runs = pd.DataFrame(rows)
    runs.to_csv(results_dir / "noise_reg_runs.csv", index=False)
    groups: dict = {}
    for j in jobs:
        groups.setdefault(j["arm"], []).append(j["name"])
    bad = find_nonconverged(results_dir, groups, factor)
    report_nonconverged(bad, factor, len(jobs) - len(missing))
    runs = runs[~runs.run_name.isin(bad)]

    order = [a for a, _, _ in ARMS]
    agg = aggregate(runs, ["arm"])
    agg["arm"] = pd.Categorical(agg.arm, order, ordered=True)
    agg = agg.sort_values("arm")
    agg.to_csv(results_dir / "noise_reg.csv", index=False)
    print(f"\n全 fp32，安静条件评估，{runs.seed.nunique()} 个 seed")
    print_agg(agg, ["arm"])

    base_auc = runs[runs.arm == "fp32"].set_index("seed").auc
    print("\n与 fp32 的同 seed 配对差：")
    for arm in order[1:]:
        d = runs[runs.arm == arm].set_index("seed").auc
        diff = (d - base_auc).dropna()
        if len(diff):
            sd = diff.std(ddof=1) if len(diff) > 1 else float("nan")
            print(f"  {arm:8s} {diff.mean():+.4f} ± {sd:.4f}   逐 seed {' '.join(f'{v:+.3f}' for v in diff)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/fast.yaml")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--shard", default=None, help="K/N：只训练第 K 份（0 起），不评估")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--converge-factor", type=float, default=2.0)
    args = ap.parse_args()

    base = load_config(args.config)
    results_dir = Path(base.output.results_dir)
    # 先跑回答两个核心问题的三组（fp32、act0.04、in0.84），强度扫描放后面：
    # 单进程串行时，中途 --eval-only 就能先看到关键结果
    first = ["fp32", "act0.04", "in0.84"]
    arms = sorted(ARMS, key=lambda a: (a[0] not in first, first.index(a[0]) if a[0] in first else 0))
    jobs = [{"arm": arm, "seed": s, "name": run_name(arm, s), "sigma_read": sr, "input_noise": inn}
            for arm, sr, inn in arms for s in args.seeds]

    if not args.eval_only:
        todo = [j for j in shard(jobs, args.shard) if not checkpoint_exists(results_dir, j["name"])]
        print(f"噪声正则化对照：共 {len(jobs)} 个任务，分片 {args.shard or '全部'} 待训 {len(todo)} 个")
        if args.dry_run:
            for j in todo:
                print(f"  {j['name']}")
            return 0
        for i, j in enumerate(todo, 1):
            print(f"\n=== [{i}/{len(todo)}] {j['name']} ===")
            try:
                params = act_params(j["sigma_read"]) if j["sigma_read"] else None
                train(job_cfg(base, j), params=params)
            except Exception as exc:  # 单个任务失败不中断整个分片
                print(f"[FAIL] {j['name']}: {type(exc).__name__}: {exc}")
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        if args.shard:
            print("\n本分片训练完成。所有分片结束后运行 --eval-only 统一评估。")
            return 0

    evaluate_all(base, results_dir, jobs, args.converge_factor)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
