"""仿真器 v2：固定满量程、饱和激活、三种读噪声模型下，报告的结论还成立吗？（NEXT-STEPS §3，2026-09-22）

对照基准是 v1.1（v1 物理 + STE 修复，`run_v11.sh`）：同一目标配置（6 tiles，深度 2），
σ_prog 3 %、σ_d2d 2 %，3 seeds × 10 颗芯片，训练与评估的物理条件一致（HWA 的前提）。

预设（每个都是在 v1.1 上叠加改动，便于逐项归因）
  fs              固定满量程（v1 噪声 σ 4 %，ReLU）                    —— 只看 A16
  fs_crelu        + clipped ReLU（饱和轨 1）                          —— 只看 A4
  fs_crelu_abs*   + 绝对噪声 σ = 0.01 / 0.03 / 0.1 / 0.3（单元满量程 U 为单位）
                  单列动态范围 DR = 20·log10(30 / σ) = 70 / 60 / 50 / 40 dB —— 交付：DR 需求
  fs_crelu_prop   + 逐元素比例噪声 σ 4 %
  fs_crelu_shot   + 散粒噪声 σ 0.02（典型隐藏层 r ≈ 0.3 时，相对噪声约 4 %）
  fs_tanh_abs0.03 tanh 代替 clipped ReLU                              —— 饱和形状
  fs_b4 / fs_b4_c / fs_c   none 架构，4 / 6 bit × 是否去直流          —— 复核 0-1 在固定满量程下是否成立

饱和轨取 1 的理由：激活前的 BN 可学习增益会抵消饱和点的缩放（见 src/models/autoencoder.py），
所以饱和本身只有和**绝对**噪声一起才有意义 —— 网络不能靠整体缩小信号躲开噪声。

短模型标定（scratchpad，20 epochs）测得 r = mean|W·x| / U：输入层约 3，隐藏层 0.15–0.6，
输出层约 11（最后一层隐藏激活 72 % 顶在饱和轨上，被用来合成 log-mel 的直流）。
瓶颈两侧 r 最小（约 0.15），在绝对噪声下最敏感。

    python scripts/sweep_v2.py --dry-run
    python scripts/sweep_v2.py --shard 0/3
    python scripts/sweep_v2.py --eval-only

产出
    results/v2_runs.csv   每 (预设, 架构, seed, 芯片) 一行
    results/v2.csv        按 (预设, 架构) 聚合，附同 seed 与 v1.1 的配对差
    results/v2_fs.csv     每个 run 的首层 D/A、末层 A/D 满量程与测试集截断率 —— 交给硬件的数字
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
from omegaconf import OmegaConf  # noqa: E402

from src.evaluate_analog import evaluate_chips  # noqa: E402
from src.experiment import (  # noqa: E402
    TARGET, aggregate, checkpoint_exists, find_nonconverged, load_model,
    print_agg, report_nonconverged, shard, target_cfg,
)
from src.features.logmel import cache_prefix, load_cache, window_start_rows  # noqa: E402
from src.models.analog import analog_layers  # noqa: E402
from src.noise import DeviceParams  # noqa: E402
from src.train import train  # noqa: E402
from src.utils import get_device, load_config  # noqa: E402

SIGMA_PROG, SIGMA_D2D = 0.03, 0.02
SEEDS = [0, 1, 2]
N_CHIPS = 10
BOTH = ("none", "per_layer")
NONE_ONLY = ("none",)

# 预设：scale / noise / act / σ / 位宽 / 去直流 / 架构
PRESETS: dict[str, dict] = {
    "fs":               dict(scale="fixed", noise="v1", act="relu", sigma=0.04, archs=BOTH),
    "fs_crelu":         dict(scale="fixed", noise="v1", act="clipped_relu", sigma=0.04, archs=BOTH),
    "fs_crelu_abs0.01": dict(scale="fixed", noise="absolute", act="clipped_relu", sigma=0.01, archs=BOTH),
    "fs_crelu_abs0.03": dict(scale="fixed", noise="absolute", act="clipped_relu", sigma=0.03, archs=BOTH),
    "fs_crelu_abs0.1":  dict(scale="fixed", noise="absolute", act="clipped_relu", sigma=0.10, archs=BOTH),
    "fs_crelu_abs0.3":  dict(scale="fixed", noise="absolute", act="clipped_relu", sigma=0.30, archs=BOTH),
    "fs_crelu_prop":    dict(scale="fixed", noise="proportional", act="clipped_relu", sigma=0.04, archs=BOTH),
    "fs_crelu_shot":    dict(scale="fixed", noise="shot", act="clipped_relu", sigma=0.02, archs=BOTH),
    "fs_tanh_abs0.03":  dict(scale="fixed", noise="absolute", act="tanh", sigma=0.03, archs=NONE_ONLY),
    "fs_b4":            dict(scale="fixed", noise="v1", act="relu", sigma=0.04, bits=4, archs=NONE_ONLY),
    "fs_b4_c":          dict(scale="fixed", noise="v1", act="relu", sigma=0.04, bits=4, center=True,
                             archs=NONE_ONLY),
    "fs_c":             dict(scale="fixed", noise="v1", act="relu", sigma=0.04, center=True, archs=NONE_ONLY),
}
V11_REF = "arch_{arch}_r0.04_d2_stefix_s{seed}"      # 同 seed 的 v1.1 对照（run_v11.sh）


def run_name(preset: str, arch: str, seed: int) -> str:
    return f"v2_{preset}_{arch}_s{seed}"


def make_params(pr: dict, arch: str) -> DeviceParams:
    b = pr.get("bits", 6)
    return DeviceParams(sigma_prog=SIGMA_PROG, sigma_d2d=SIGMA_D2D, sigma_read=pr["sigma"],
                        adc_mode=arch, scale_mode=pr["scale"], noise_model=pr["noise"],
                        w_bits=b, s_bits=b, b_bits=b)


def job_cfg(base, pr: dict, name: str, seed: int):
    cfg = target_cfg(base, name, seed)
    return OmegaConf.merge(cfg, {"model": {"act": pr["act"]},
                                 "feature": {"center": bool(pr.get("center", False))}})


def build_jobs(presets: list[str], seeds: list[int]) -> list[dict]:
    return [{"preset": p, "arch": a, "seed": s, "name": run_name(p, a, s)}
            for p in presets for a in PRESETS[p]["archs"] for s in seeds]


def fs_deliverables(model, j: dict, test_x: torch.Tensor) -> dict:
    """首层 D/A、末层 A/D 满量程（log-mel dB 单位，去直流时是去直流后的量），以及测试集截断率。"""
    layers = analog_layers(model)
    x = test_x - model.input_offset.cpu() if getattr(model, "center", False) else test_x
    d_a, a_d = float(layers[0].in_fs), float(layers[-1].out_fs)
    return {"preset": j["preset"], "arch": j["arch"], "seed": j["seed"],
            "da_full_scale": d_a, "ad_full_scale": a_d,
            "da_clip_rate": float((x.abs() > d_a).float().mean()) if d_a > 0 else float("nan")}


def evaluate_all(base, results_dir: Path, jobs: list[dict], factor: float) -> None:
    device = get_device()
    mel, meta = load_cache(cache_prefix(base.data.cache_dir, TARGET["n_mels"], "test"))
    starts = window_start_rows(meta, TARGET["frames"])
    pick = np.random.default_rng(0).choice(starts, 20000, replace=False)
    test_x = torch.from_numpy(np.stack([mel[i:i + TARGET["frames"]].reshape(-1) for i in pick])
                              .astype(np.float32))
    mids, max_fpr = list(base.eval.machine_ids), float(base.eval.max_fpr)

    rows, fs_rows, missing = [], [], []
    t0 = time.time()
    for j in jobs:
        if not checkpoint_exists(results_dir, j["name"]):
            missing.append(j["name"])
            continue
        pr = PRESETS[j["preset"]]
        p = make_params(pr, j["arch"])
        model = load_model(job_cfg(base, pr, j["name"], j["seed"]), results_dir, j["name"], device, p)
        fs_rows.append(fs_deliverables(model, j, test_x))
        df = evaluate_chips(model, mel, meta, TARGET["frames"], device, p,
                            n_chips=N_CHIPS, machine_ids=mids, max_fpr=max_fpr)
        rows += [{"preset": j["preset"], "arch": j["arch"], "seed": j["seed"],
                  "run_name": j["name"], **r} for r in df.to_dict("records")]
    print(f"\n评估 {len(jobs) - len(missing)} 个模型，用时 {time.time() - t0:.0f}s")
    if missing:
        print(f"[注意] {len(missing)} 个 checkpoint 缺失：{', '.join(missing[:8])}"
              f"{' ...' if len(missing) > 8 else ''}")
    if not rows:
        return

    runs = pd.DataFrame(rows)
    runs.to_csv(results_dir / "v2_runs.csv", index=False)
    pd.DataFrame(fs_rows).to_csv(results_dir / "v2_fs.csv", index=False)

    groups: dict = {}
    for j in jobs:
        groups.setdefault((j["preset"], j["arch"]), []).append(j["name"])
    bad = find_nonconverged(results_dir, groups, factor)
    report_nonconverged(bad, factor, len(jobs) - len(missing))
    runs = runs[~runs.run_name.isin(bad)]

    agg = aggregate(runs, ["preset", "arch"])
    # 同 seed 与 v1.1 的配对差（先对芯片取平均）
    ref_f = results_dir / "arch_stefix_runs.csv"
    if ref_f.exists():
        ref = (pd.read_csv(ref_f).query("depth == 2 and sigma_read == 0.04")
               .groupby(["arch", "seed"]).auc.mean())
        per = runs.groupby(["preset", "arch", "seed"]).auc.mean()
        diffs = []
        for (preset, arch), g in per.groupby(level=["preset", "arch"]):
            d = [g[(preset, arch, s)] - ref[(arch, s)] for s in g.index.get_level_values("seed")
                 if (arch, s) in ref.index]
            diffs.append({"preset": preset, "arch": arch,
                          "vs_v11_mean": float(np.mean(d)) if d else np.nan,
                          "vs_v11_std": float(np.std(d, ddof=1)) if len(d) > 1 else np.nan})
        agg = agg.merge(pd.DataFrame(diffs), on=["preset", "arch"], how="left")
    agg["preset"] = pd.Categorical(agg.preset, list(PRESETS), ordered=True)
    agg = agg.sort_values(["preset", "arch"])
    agg.to_csv(results_dir / "v2.csv", index=False)

    bar = "=" * 96
    print(f"\n{bar}\n结果 -> {results_dir / 'v2.csv'}\n{bar}")
    print_agg(agg, ["preset", "arch"])
    if "vs_v11_mean" in agg:
        print("\n与 v1.1（同 seed，none / per_layer 各自对应）的配对差：")
        for r in agg.itertuples():
            print(f"  {r.preset:<18} {r.arch:<10} {r.vs_v11_mean:+.4f} ± {r.vs_v11_std:.4f}")
    print("\nnone − per_layer（同 seed 配对）：")
    per = runs.groupby(["preset", "arch", "seed"]).auc.mean().unstack("arch")
    if set(BOTH) <= set(per.columns):
        d = (per["none"] - per["per_layer"]).dropna().groupby(level="preset").agg(["mean", "std"])
        for preset, r in d.iterrows():
            print(f"  {preset:<18} {r['mean']:+.4f} ± {r['std']:.4f}")
    fs = pd.DataFrame(fs_rows).groupby(["preset", "arch"])[
        ["da_full_scale", "ad_full_scale", "da_clip_rate"]].mean().round(4)
    print("\n转换器满量程（log-mel dB；去直流的预设是去直流之后的量）与 D/A 截断率：")
    print(fs.to_string())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/fast.yaml")
    ap.add_argument("--presets", nargs="+", default=list(PRESETS), choices=list(PRESETS))
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--shard", default=None, help="K/N：只训练第 K 份（0 起），不评估")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--converge-factor", type=float, default=2.0)
    args = ap.parse_args()

    base = load_config(args.config)
    results_dir = Path(base.output.results_dir)
    jobs = build_jobs(args.presets, args.seeds)

    if not args.eval_only:
        mine = shard(jobs, args.shard)
        todo = [j for j in mine if not checkpoint_exists(results_dir, j["name"])]
        print(f"仿真器 v2：共 {len(jobs)} 个训练任务；分片 {args.shard or '全部'} 分到 {len(mine)} 个，"
              f"待训 {len(todo)} 个")
        if args.dry_run:
            for j in todo:
                print(f"  {j['name']}")
            return 0
        for i, j in enumerate(todo, 1):
            print(f"\n=== [{i}/{len(todo)}] {j['name']} ===")
            pr = PRESETS[j["preset"]]
            try:
                train(job_cfg(base, pr, j["name"], j["seed"]), params=make_params(pr, j["arch"]))
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
