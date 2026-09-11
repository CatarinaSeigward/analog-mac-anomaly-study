"""Day 4：比特扫描 → 图 4。（审稿后定稿，替代草稿版本）

============================================================================
审稿结论
============================================================================
1. 原草稿按 (bits × σ 分组) 训练会是 45 次，而且 σ 分组里的 0 与 10% 和训练
   条件不匹配 —— HWA 的前提是训练与部署匹配。
   → 每个位宽只在标称噪声下做匹配 QAT；同一个 checkpoint 再在"关掉全部噪声"
     条件下评估一次，得到纯量化误差的参考线。
2. 必须在修正后的 Ai Linear 架构（adc_mode="none"）上跑。该架构下 S（信号）
   位宽只作用于第一层输入 D/A 与最后一层输出 A/D，W / B 位宽作用于每一层。
3. W / S / B 三者同步变化（危险点 F）；--sweep-only w|s|b 做单项敏感度。
4. 用 QAT 而不是 PTQ：每个位宽各自训练。bits=2 即对称三值 {-1, 0, +1}·scale。

============================================================================
训练配方（Day 4 结果后更新）
============================================================================
第一次运行用 σ_read = 0 训练，结果被一个配方问题污染：none 架构在训练时
没有激活噪声，得到的模型对器件噪声特别敏感（6-bit 下 0.654 ± 0.044）。
交叉评估（scripts/probe_train_noise.py）表明决定因素是**训练时**的激活噪声：
σ_read = 4% 训练的模型部署到安静芯片上仍有 0.738 ± 0.014。
→ 默认 --train-sigma-read 0.04。原运行可用 --train-sigma-read 0 复现，
  其 checkpoint 与 results/bits.csv 保留不动（命名不带 _r 后缀）。

============================================================================
运行
============================================================================
  python scripts/sweep_bits.py --dry-run
  python scripts/sweep_bits.py --shard 0/4        # 多进程并行：只训练本分片
  python scripts/sweep_bits.py --eval-only        # 全部分片结束后统一评估聚合

产出
  results/bits_r0.04_runs.csv / results/bits_r0.04.csv    （--train-sigma-read 0 时为 bits_runs.csv / bits.csv）
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

BITS = [2, 4, 6, 8, 10]
CHIP_BITS = 6                              # 芯片实际规格，图上要标竖线
SIGMA_PROG, SIGMA_D2D = 0.03, 0.02         # 标称噪声（Day 3 推荐容差）
TRAIN_SIGMA_READ = 0.04                    # 训练配方：none 架构需要训练时的激活噪声
ADC_MODE = "none"                          # Ai Linear 架构
SEEDS = [0, 1, 2]
N_CHIPS = 10
MIN_PER_RUN = 6.9                          # Day 3 实测（深度 2，单进程）


def suffix(train_sr: float) -> str:
    """σ_read = 0 时不加后缀，保持第一次运行的命名不变。"""
    return "" if train_sr == 0 else f"_r{train_sr:g}"


def bit_kwargs(bits: int, only: str | None) -> dict:
    if only is None:
        return {"w_bits": bits, "s_bits": bits, "b_bits": bits}
    kw = {"w_bits": CHIP_BITS, "s_bits": CHIP_BITS, "b_bits": CHIP_BITS}
    kw[f"{only}_bits"] = bits
    return kw


def run_name(bits: int, seed: int, only: str | None, train_sr: float) -> str:
    return f"bits_{only or 'wsb'}{bits}{suffix(train_sr)}_s{seed}"


def make_params(bits: int, only: str | None, sigma_prog: float, sigma_d2d: float,
                sigma_read: float) -> DeviceParams:
    return DeviceParams(sigma_prog=sigma_prog, sigma_d2d=sigma_d2d, sigma_read=sigma_read,
                        adc_mode=ADC_MODE, **bit_kwargs(bits, only))


def conditions(train_sr: float) -> dict[str, tuple[float, float, float]]:
    """评估条件 (σ_prog, σ_d2d, σ_read)。nominal 与训练匹配；quant_only 关掉全部噪声。"""
    return {"nominal": (SIGMA_PROG, SIGMA_D2D, train_sr), "quant_only": (0.0, 0.0, 0.0)}


def evaluate_all(base, results_dir: Path, jobs: list[dict], only: str | None,
                 train_sr: float, n_chips: int, factor: float) -> None:
    device = get_device()
    mel, meta = load_cache(cache_prefix(base.data.cache_dir, TARGET["n_mels"], "test"))
    mids, max_fpr = list(base.eval.machine_ids), float(base.eval.max_fpr)
    conds = conditions(train_sr)

    rows, missing = [], []
    t0 = time.time()
    for j in jobs:
        if not checkpoint_exists(results_dir, j["name"]):
            missing.append(j["name"])
            continue
        cfg = target_cfg(base, j["name"], j["seed"])
        model = load_model(cfg, results_dir, j["name"], device,
                           make_params(j["bits"], only, *conds["nominal"]))
        for cond, noise in conds.items():
            p = make_params(j["bits"], only, *noise)
            chips = n_chips if any(v > 0 for v in noise) else 1   # 无噪声时各芯片完全相同
            df = evaluate_chips(model, mel, meta, TARGET["frames"], device, p,
                                n_chips=chips, machine_ids=mids, max_fpr=max_fpr)
            rows += [{"bits": j["bits"], "condition": cond, "seed": j["seed"],
                      "run_name": j["name"], **r} for r in df.to_dict("records")]

    print(f"\n评估 {len(jobs) - len(missing)} 个模型，用时 {time.time() - t0:.0f}s")
    if missing:
        more = " ..." if len(missing) > 6 else ""
        print(f"[注意] {len(missing)} 个 checkpoint 缺失，未纳入：{', '.join(missing[:6])}{more}")
    if not rows:
        return

    tag = suffix(train_sr)
    runs = pd.DataFrame(rows)
    runs.to_csv(results_dir / f"bits{tag}_runs.csv", index=False)

    groups: dict = {}
    for j in jobs:
        groups.setdefault(j["bits"], []).append(j["name"])
    bad = find_nonconverged(results_dir, groups, factor)
    report_nonconverged(bad, factor, len(jobs) - len(missing))
    runs = runs[~runs.run_name.isin(bad)]

    keys = ["condition", "bits"]
    agg = aggregate(runs, keys)
    agg.to_csv(results_dir / f"bits{tag}.csv", index=False)
    bar = "=" * 96
    print(f"\n{bar}\n结果 -> {results_dir / f'bits{tag}.csv'}   (训练 σ_read = {train_sr:g})\n{bar}")
    print_agg(agg, keys)

    nom = agg[agg.condition == "nominal"].set_index("bits")
    if CHIP_BITS in nom.index:
        print(f"\n相对芯片规格 {CHIP_BITS}-bit 的 AUC 差（nominal 条件）：")
        for b in nom.index:
            print(f"  {b:>2}-bit  {nom.auc_mean[b] - nom.auc_mean[CHIP_BITS]:+.4f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/fast.yaml")
    ap.add_argument("--bits", type=int, nargs="+", default=BITS)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--n-chips", type=int, default=N_CHIPS)
    ap.add_argument("--train-sigma-read", type=float, default=TRAIN_SIGMA_READ,
                    help="训练（及 nominal 评估）时的读噪声。0 = 复现第一次运行")
    ap.add_argument("--sweep-only", choices=["w", "s", "b"], default=None,
                    help="只扫某一项位宽，其余固定在 6-bit（敏感度分析用）")
    ap.add_argument("--shard", default=None, help="K/N：只训练第 K 份（0 起），不评估")
    ap.add_argument("--eval-only", action="store_true", help="不训练，评估已有 checkpoint 并聚合")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--converge-factor", type=float, default=2.0)
    args = ap.parse_args()

    base = load_config(args.config)
    results_dir = Path(base.output.results_dir)
    only, train_sr = args.sweep_only, args.train_sigma_read
    jobs = [{"bits": b, "seed": s, "name": run_name(b, s, only, train_sr)}
            for b in args.bits for s in args.seeds]

    if not args.eval_only:
        mine = shard(jobs, args.shard)
        todo = [j for j in mine if args.force or not checkpoint_exists(results_dir, j["name"])]
        print(f"比特扫描（{'W/S/B 同步' if only is None else only + ' 单项'}，QAT，"
              f"adc_mode={ADC_MODE}，训练 σ_read={train_sr:g}）："
              f"共 {len(jobs)} 个训练任务；分片 {args.shard or '全部'} 分到 {len(mine)} 个，"
              f"待训 {len(todo)} 个（单进程约 {len(todo) * MIN_PER_RUN:.0f} 分钟）")
        if args.dry_run:
            for j in todo:
                print(f"  {j['name']}")
            return 0
        for i, j in enumerate(todo, 1):
            print(f"\n=== [{i}/{len(todo)}] {j['name']} ===")
            try:
                cfg = target_cfg(base, j["name"], j["seed"])
                train(cfg, params=make_params(j["bits"], only, *conditions(train_sr)["nominal"]))
            except Exception as exc:  # 单个任务失败不中断整个分片
                print(f"[FAIL] {j['name']}: {type(exc).__name__}: {exc}")
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        if args.shard:
            print("\n本分片训练完成。所有分片结束后运行 --eval-only 统一评估。")
            return 0

    evaluate_all(base, results_dir, jobs, only, train_sr, args.n_chips, args.converge_factor)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
