"""Day 4：层边界 A/D 放置 × 信号通路读噪声 → 图 3。

替代审稿前的草稿 ``sweep_depth.py``。

============================================================================
审稿结论（Day 4 开跑前）
============================================================================
1. ★ Day 3 的 ``AnalogLinear`` 在**每一层的输入**都做 6-bit 量化 —— 相当于每个
   层边界都有一次 D/A。Day 3 标为 "C3 / Ai Linear 建模" 的配置因此从未仿真过
   "隐藏层无 A/D、D/A"；C3 vs C4 比的只是"每个边界量化 1 次 vs 2 次"。
   → Day 3 结论 6 撤回。新增 ``DeviceParams.adc_mode``，按层位置放置 A/D、D/A
     （``models.analog.apply_adc_mode``），并加了回归测试。

2. 即使放置正确，σ_read = 0 时信号通路上没有加性噪声，层间 A/D 没有东西可以
   "再生"；而乘性权重噪声在 σ_prog ≥ 3% 时远大于 1 LSB，量化消除不了。
   → 主自变量改为 **σ_read（信号通路读噪声）** —— 这才是 A/D 能起作用的地方。

3. 草稿的"同一 checkpoint、两种推理方式"在放置修正后不再成立：两种架构的量化点
   完全不同，一个模型只能与其中一种匹配，而 HWA 的前提就是训练与部署匹配。
   → 两种架构各自做匹配的 HWA 训练，按种子配对比较。

4. 危险点 K 保留：深度扫描中每层固定 30 宽以匹配 tile，更深 = 更多参数，
   深度与容量是混杂的。图注必须声明。

============================================================================
设计
============================================================================
  两种架构（DeviceParams.adc_mode）
    none       仅第一层输入 D/A、最后一层输出 A/D，隐藏层全程模拟  —— Ai Linear
    per_layer  每层输入 D/A + 输出 A/D                             —— 常规 CIM

  Part A（图 3A，主结果）  深度 2（目标配置），σ_read ∈ SIGMA_READS
  Part B（图 3B，深度）    深度 ∈ {1, 2, 4}，σ_read ∈ DEPTH_SIGMA_READS
  两部分都固定 σ_prog = 3%（Day 3：HWA 下 ≤3% 基本免费）、σ_d2d = 2%。

  读噪声 = 加性高斯，std = σ_read × mean|W·x|（不含偏置，逐层按信号量级归一）。
  换算到 A/D 的 LSB：噪声/LSB ≈ 31 · σ_read · mean|W·x| / max|y|。
  SIGMA_READS 按实测的 mean|W·x|/max|y|（均值 0.191）反推，
  噪声/LSB ≈ 5.93 × σ_read，取值覆盖 0 / 0.25 / 0.5 / 1 / 2 LSB。
  ★ 这是为了让 A/D 的"再生"效应可观测而选的**压力区间，不是实测器件参数**
    （assumptions.md A15）。

============================================================================
运行
============================================================================
  python scripts/sweep_arch.py --dry-run
  python scripts/sweep_arch.py --part A --shard 0/3      # 多进程并行：只训练本分片
  python scripts/sweep_arch.py --eval-only               # 全部分片结束后统一评估聚合

产出
  results/arch_runs.csv   每 (架构, σ_read, 深度, 种子, 芯片) 一行
  results/arch.csv        按 (深度, σ_read, 架构) 聚合
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.evaluate_analog import evaluate_chips  # noqa: E402
from src.experiment import (  # noqa: E402
    TARGET, aggregate, checkpoint_exists, find_nonconverged, load_model,
    print_agg, report_nonconverged, shard, target_cfg,
)
from src.features.logmel import cache_prefix, load_cache  # noqa: E402
from src.noise import DeviceParams  # noqa: E402
from src.train import train  # noqa: E402
from src.utils import get_device, load_config  # noqa: E402

SIGMA_PROG = 0.03
SIGMA_D2D = 0.02
ARCHS = ("none", "per_layer")            # Ai Linear vs 常规 CIM

# Part A：实测 mean|W·x|/max|y| 的均值为 0.191（深度 2 的 HWA 模型，各层 0.07–0.44），
# 即 噪声/LSB ≈ 5.93 × σ_read。下列取值对应 噪声/LSB ≈ 0 / 0.24 / 0.47 / 1.0 / 2.0。
SIGMA_READS = [0.0, 0.04, 0.08, 0.17, 0.34]
DEPTH_SIGMA_READS = [0.0, 0.08]          # Part B：无读噪声 + 约 0.5 LSB（A/D 再生最该起作用处）
DEPTHS_B = [1, 4]                        # 深度 2 的参照直接取自 Part A，不重复训练
SEEDS = [0, 1, 2]
N_CHIPS = 10

MIN_PER_LAYER = 1.15                     # Day 3 实测：6 层约 6.9 分钟（kernel 启动主导）


def layer_count(depth: int) -> int:
    return 2 * depth + 2


def run_name(arch: str, sigma_read: float, depth: int, seed: int) -> str:
    return f"arch_{arch}_r{sigma_read:g}_d{depth}_s{seed}"


def make_params(arch: str, sigma_read: float) -> DeviceParams:
    return DeviceParams(sigma_prog=SIGMA_PROG, sigma_d2d=SIGMA_D2D,
                        sigma_read=sigma_read, adc_mode=arch)


def build_jobs(parts, seeds, sigma_reads, depth_sigma_reads, depths_b) -> list[dict]:
    jobs, seen = [], set()

    def add(part, arch, sr, depth, seed):
        name = run_name(arch, sr, depth, seed)
        if name not in seen:
            seen.add(name)
            jobs.append({"part": part, "arch": arch, "sigma_read": sr,
                         "depth": depth, "seed": seed, "name": name})

    if "A" in parts:
        for sr in sigma_reads:
            for arch in ARCHS:
                for s in seeds:
                    add("A", arch, sr, 2, s)
    if "B" in parts:
        # 只训练额外的深度；深度 2 的参照属于 Part A。若 Part B 单独运行时也把
        # 深度 2 算进来，会与仍在训练 Part A 的并行进程写同一个 checkpoint 目录。
        for sr in depth_sigma_reads:
            for d in depths_b:
                for arch in ARCHS:
                    for s in seeds:
                        add("B", arch, sr, d, s)
    return jobs


def paired_diff(runs: pd.DataFrame, depth: int) -> pd.DataFrame | None:
    """同一种子下 none − per_layer 的 AUC 差（先对芯片取平均），再跨种子汇总。"""
    d = runs[runs.depth == depth]
    per = d.groupby(["sigma_read", "arch", "seed"]).auc.mean().unstack("arch")
    if not set(ARCHS) <= set(per.columns):
        return None
    per = per.dropna()
    per["diff"] = per["none"] - per["per_layer"]
    return per.groupby("sigma_read")["diff"].agg(["mean", "std", "count"])


def evaluate_all(base, results_dir: Path, jobs: list[dict], n_chips: int, factor: float) -> None:
    device = get_device()
    mel, meta = load_cache(cache_prefix(base.data.cache_dir, TARGET["n_mels"], "test"))
    mids, max_fpr = list(base.eval.machine_ids), float(base.eval.max_fpr)

    rows, missing = [], []
    t0 = time.time()
    for j in jobs:
        if not checkpoint_exists(results_dir, j["name"]):
            missing.append(j["name"])
            continue
        cfg = target_cfg(base, j["name"], j["seed"], n_blocks=j["depth"])
        p = make_params(j["arch"], j["sigma_read"])
        model = load_model(cfg, results_dir, j["name"], device, p)
        df = evaluate_chips(model, mel, meta, TARGET["frames"], device, p,
                            n_chips=n_chips, machine_ids=mids, max_fpr=max_fpr)
        rows += [{"arch": j["arch"], "sigma_read": j["sigma_read"], "depth": j["depth"],
                  "seed": j["seed"], "run_name": j["name"], **r}
                 for r in df.to_dict("records")]

    print(f"\n评估 {len(jobs) - len(missing)} 个模型 × {n_chips} 颗芯片，用时 {time.time() - t0:.0f}s")
    if missing:
        more = " ..." if len(missing) > 6 else ""
        print(f"[注意] {len(missing)} 个 checkpoint 缺失，未纳入：{', '.join(missing[:6])}{more}")
    if not rows:
        return

    runs = pd.DataFrame(rows)
    runs.to_csv(results_dir / "arch_runs.csv", index=False)

    groups: dict = {}
    for j in jobs:
        groups.setdefault((j["arch"], j["sigma_read"], j["depth"]), []).append(j["name"])
    bad = find_nonconverged(results_dir, groups, factor)
    report_nonconverged(bad, factor, len(jobs) - len(missing))
    runs = runs[~runs.run_name.isin(bad)]

    keys = ["depth", "sigma_read", "arch"]
    agg = aggregate(runs, keys)
    agg.to_csv(results_dir / "arch.csv", index=False)
    bar = "=" * 96
    print(f"\n{bar}\n结果 -> {results_dir / 'arch.csv'}\n{bar}")
    print_agg(agg, keys)

    for depth in sorted(runs.depth.unique()):
        pdiff = paired_diff(runs, depth)
        if pdiff is None:
            continue
        print(f"\n深度 {depth}：none − per_layer（同种子配对，AUC 差）")
        for sr, r in pdiff.iterrows():
            std = 0.0 if pd.isna(r["std"]) else r["std"]
            print(f"  σ_read={sr:<5g} {r['mean']:+.4f} ± {std:.4f}  (n={int(r['count'])} 种子)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/fast.yaml")
    ap.add_argument("--part", nargs="+", choices=["A", "B"], default=["A", "B"])
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--sigma-reads", type=float, nargs="+", default=SIGMA_READS)
    ap.add_argument("--depth-sigma-reads", type=float, nargs="+", default=DEPTH_SIGMA_READS)
    ap.add_argument("--depths-b", type=int, nargs="+", default=DEPTHS_B)
    ap.add_argument("--n-chips", type=int, default=N_CHIPS)
    ap.add_argument("--shard", default=None, help="K/N：只训练第 K 份（0 起），不评估")
    ap.add_argument("--eval-only", action="store_true", help="不训练，评估已有 checkpoint 并聚合")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--converge-factor", type=float, default=2.0)
    args = ap.parse_args()

    base = load_config(args.config)
    results_dir = Path(base.output.results_dir)
    jobs = build_jobs(args.part, args.seeds, args.sigma_reads,
                      args.depth_sigma_reads, args.depths_b)
    missing_ref = sorted(set(args.depth_sigma_reads) - set(args.sigma_reads))
    if "B" in args.part and missing_ref:
        print(f"[警告] σ_read={missing_ref} 不在 Part A 的网格里，"
              f"Part B 在这些取值下缺少深度 2 的参照")

    if not args.eval_only:
        order = sorted(jobs, key=lambda j: (-layer_count(j["depth"]), j["name"]))
        mine = shard(order, args.shard)
        todo = [j for j in mine if args.force or not checkpoint_exists(results_dir, j["name"])]
        est = sum(layer_count(j["depth"]) for j in todo) * MIN_PER_LAYER
        print(f"Part {'+'.join(args.part)}：共 {len(jobs)} 个训练任务；"
              f"分片 {args.shard or '全部'} 分到 {len(mine)} 个，待训 {len(todo)} 个"
              f"（单进程约 {est:.0f} 分钟）")
        if args.dry_run:
            for j in todo:
                print(f"  {j['name']}")
            return 0
        for i, j in enumerate(todo, 1):
            print(f"\n=== [{i}/{len(todo)}] {j['name']} ===")
            try:
                cfg = target_cfg(base, j["name"], j["seed"], n_blocks=j["depth"])
                train(cfg, params=make_params(j["arch"], j["sigma_read"]))
            except Exception as exc:  # 单个任务失败不中断整个分片
                print(f"[FAIL] {j['name']}: {type(exc).__name__}: {exc}")
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        if args.shard:
            print("\n本分片训练完成。所有分片结束后运行 --eval-only 统一评估。")
            return 0

    evaluate_all(base, results_dir, jobs, args.n_chips, args.converge_factor)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
