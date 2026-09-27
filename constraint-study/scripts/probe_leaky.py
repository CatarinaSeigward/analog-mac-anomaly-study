"""NEXT-STEPS 0-4：用 leaky integrator 代替多帧拼接，时间上下文还在不在？（2026-09-22）

背景
    报告结论 2：同样 30 维输入，6 bands × 5 frames 比 30 bands × 1 frame 高 0.068 AUC。
    报告据此建议前端用"几个可配置时间常数的 leaky integrator"提供时间上下文，
    但实验里测的是 5 帧拼接 —— 那需要把过去的帧存下来，而 leaky integrator 只保留一个状态。
    这条建议本身没有被测过。

设计
    6 个 mel band，每个 band 过 5 个一阶 IIR（EMA），同一时刻的 30 个输出作为一个输入（frames = 1）：
        y[n] = α · y[n-1] + (1 - α) · x[n]，  α = exp(-hop / τ)，hop = 32 ms
    每个文件开头按稳态初始化（y[-1] = x[0]），不跨文件。维度与 6 × 5 帧相同（30），不混入维度变量。
      tau512   τ = 32 / 64 / 128 / 256 / 512 ms
      tau128   τ = 32 / 45 / 64 / 91 / 128 ms   —— 最长 τ 缩短到 1/4：最长需要多大的 τ？
    ⚠️ 在 log 域积分，是一阶近似；真实电路是先整流积分、再取 log（NEXT-STEPS §5B 的 B2）。

    对照：同配置的 6 × 5 帧，同 seed 配对。
      默认（第一轮）  fp32 两组 τ + hwa6 的 tau512；帧拼接基准取已有结果
                      （fp32：sweep_runs.csv；HWA：STE 修复后的 6-bit 位宽扫描）
      --lr / --tag    受控重跑：指定学习率，帧拼接基准也用同一学习率重训（--arms 里含 fp32:frames）

    第一轮发现 EMA 特征的验证损失全程有尖峰（lr 0.002），最终 epoch 可能落在尖峰上，
    所以每个 run 另记最低与最终验证损失。

    python scripts/probe_leaky.py
    python scripts/probe_leaky.py --lr 0.0005 --tag lr5e-4 --arms fp32:frames fp32:tau512 fp32:tau128

产出
    cache/ema_<tag>/n_mels30/{train,test}_logmel.npy
    results/leaky[_<tag>]_runs.csv / results/leaky[_<tag>].csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from scipy.signal import lfilter, lfilter_zi  # noqa: E402

from src.evaluate_analog import evaluate_chips  # noqa: E402
from src.experiment import TARGET, checkpoint_exists, load_model, target_cfg  # noqa: E402
from src.features.logmel import cache_prefix, load_cache  # noqa: E402
from src.noise import IDEAL, DeviceParams  # noqa: E402
from src.train import train  # noqa: E402
from src.utils import get_device, load_config  # noqa: E402

HOP_MS = 1000.0 * 512 / 16000          # 32 ms
TAUS = {"tau512": [32, 64, 128, 256, 512], "tau128": [32, 45, 64, 91, 128]}
SEEDS = [0, 1, 2]
N_CHIPS = 10
NOMINAL = DeviceParams(sigma_prog=0.03, sigma_d2d=0.02, sigma_read=0.04, adc_mode="none")
# (训练方式, 特征)。特征 "frames" = 原来的 6 × 5 帧拼接
ARMS = [("fp32", "tau512"), ("fp32", "tau128"), ("hwa6", "tau512")]


def ema_dir(base, tag: str) -> Path:
    return Path(base.data.cache_dir) / f"ema_{tag}"


def build_ema_cache(base, tag: str) -> None:
    """从 6 band 的 log-mel 缓存算出 [n_frames, 5 τ × 6 band] 的 EMA 特征。"""
    alphas = [float(np.exp(-HOP_MS / t)) for t in TAUS[tag]]
    for split in ("train", "test"):
        out_prefix = cache_prefix(ema_dir(base, tag), 30, split)
        if Path(f"{out_prefix}_logmel.npy").exists():
            continue
        mel, meta = load_cache(cache_prefix(base.data.cache_dir, 6, split))
        out = np.empty((mel.shape[0], 30), dtype=np.float32)
        for e in meta["files"]:
            seg = np.asarray(mel[e["offset"]: e["offset"] + e["n_frames"]], dtype=np.float64)
            for k, a in enumerate(alphas):
                b_, a_ = [1.0 - a], [1.0, -a]
                zi = lfilter_zi(b_, a_)[:, None] * seg[0][None, :]    # 稳态初始化：y[-1] = x[0]
                y, _ = lfilter(b_, a_, seg, axis=0, zi=zi)
                out[e["offset"]: e["offset"] + e["n_frames"], 6 * k: 6 * (k + 1)] = y
        out_prefix.parent.mkdir(parents=True, exist_ok=True)
        np.save(f"{out_prefix}_logmel.npy", out)
        meta = {**meta, "n_mels": 30, "ema_taus_ms": TAUS[tag], "source": "n_mels6 EMA"}
        with open(f"{out_prefix}_meta.json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh)
        print(f"[cache] {out_prefix}  {out.shape}")


def run_name(kind: str, feat: str, seed: int, run_tag: str | None = None) -> str:
    return f"leaky_{feat}_{kind}{'_' + run_tag if run_tag else ''}_s{seed}"


def arm_cfg(base, kind: str, feat: str, seed: int, lr: float | None, run_tag: str | None):
    cfg = target_cfg(base, run_name(kind, feat, seed, run_tag), seed)
    if feat != "frames":
        cfg = OmegaConf.merge(cfg, {"data": {"cache_dir": str(ema_dir(base, feat))},
                                    "feature": {"n_mels": 30, "frames": 1}})
    if lr is not None:
        cfg = OmegaConf.merge(cfg, {"train": {"lr": lr}})
    return cfg


def test_cache(base, feat: str):
    if feat == "frames":
        return (*load_cache(cache_prefix(base.data.cache_dir, TARGET["n_mels"], "test")),
                TARGET["frames"])
    return (*load_cache(cache_prefix(ema_dir(base, feat), 30, "test")), 1)


def val_losses(results_dir: Path, name: str) -> tuple[float, float]:
    h = json.loads((results_dir / name / "history.json").read_text(encoding="utf-8"))["history"]
    v = [e["val_loss"] for e in h]
    return min(v), v[-1]


def old_baselines(results_dir: Path) -> pd.DataFrame:
    """第一轮的 6 × 5 帧对照：fp32 来自 sweep_runs.csv，HWA 来自 STE 修复后的 6-bit 位宽扫描。"""
    sw = pd.read_csv(results_dir / "sweep_runs.csv").query("cfg_name == 'm6f5_h30b2z4'")
    rows = [{"kind": "fp32", "features": "frames", "seed": int(r.seed), "auc": r.auc, "pauc": r.pauc}
            for r in sw.itertuples()]
    bf = results_dir / "bits_r0.04_b2-4-5-6-8-10_stefix_runs.csv"
    if bf.exists():
        b = pd.read_csv(bf).query("bits == 6 and condition == 'nominal'")
        for seed, g in b.groupby("seed"):
            rows.append({"kind": "hwa6", "features": "frames", "seed": int(seed),
                         "auc": g.auc.mean(), "pauc": g.pauc.mean()})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/fast.yaml")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--lr", type=float, default=None, help="覆盖 train.lr（受控重跑用）")
    ap.add_argument("--tag", default=None, help="加进 run 名与结果文件名；给了 --lr 时必须给")
    ap.add_argument("--arms", nargs="+", default=None, help="如 fp32:frames fp32:tau512 hwa6:tau512")
    args = ap.parse_args()
    if args.lr is not None and not args.tag:
        ap.error("--lr 会改变训练，必须配 --tag，否则会与第一轮的 run 重名")
    arms = [tuple(a.split(":")) for a in args.arms] if args.arms else ARMS

    base = load_config(args.config)
    results_dir = Path(base.output.results_dir)
    device = get_device()
    for feat in {f for _, f in arms if f != "frames"}:
        build_ema_cache(base, feat)

    if not args.eval_only:
        for kind, feat in arms:
            for seed in SEEDS:
                name = run_name(kind, feat, seed, args.tag)
                if checkpoint_exists(results_dir, name):
                    continue
                print(f"\n=== {name} ===")
                try:
                    train(arm_cfg(base, kind, feat, seed, args.lr, args.tag),
                          params=NOMINAL if kind == "hwa6" else None)
                except Exception as exc:
                    print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
                finally:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    rows = []
    for kind, feat in arms:
        mel, meta, frames = test_cache(base, feat)
        p = NOMINAL if kind == "hwa6" else IDEAL
        for seed in SEEDS:
            name = run_name(kind, feat, seed, args.tag)
            if not checkpoint_exists(results_dir, name):
                print(f"[缺] {name}")
                continue
            model = load_model(arm_cfg(base, kind, feat, seed, args.lr, args.tag),
                               results_dir, name, device, p)
            df = evaluate_chips(model, mel, meta, frames, device, p,
                                n_chips=N_CHIPS if kind == "hwa6" else 1,
                                machine_ids=list(base.eval.machine_ids),
                                max_fpr=float(base.eval.max_fpr))
            vmin, vfin = val_losses(results_dir, name)
            rows.append({"kind": kind, "features": feat, "seed": seed, "auc": df.auc.mean(),
                         "pauc": df.pauc.mean(), "auc_min": df.auc.min(),
                         "val_min": vmin, "val_final": vfin})
    runs = pd.DataFrame(rows)
    # 本轮没有重训帧拼接基准的训练方式，用第一轮的已有结果当对照
    old = old_baselines(results_dir)
    trained_frames = {k for k, f in arms if f == "frames"}
    runs = pd.concat([runs, old[~old.kind.isin(trained_frames)
                                & old.kind.isin({k for k, _ in arms})]], ignore_index=True)
    suffix = f"_{args.tag}" if args.tag else ""
    runs.to_csv(results_dir / f"leaky{suffix}_runs.csv", index=False)

    agg = (runs.groupby(["kind", "features"]).agg(n_seeds=("seed", "nunique"),
                                                  auc_mean=("auc", "mean"), auc_std=("auc", "std"),
                                                  pauc_mean=("pauc", "mean"),
                                                  val_min=("val_min", "mean"),
                                                  val_final=("val_final", "mean"))
           .round(4).reset_index())
    agg.to_csv(results_dir / f"leaky{suffix}.csv", index=False)
    lr_note = f"lr {args.lr:g}" if args.lr is not None else "lr 按配置"
    print(f"\n时间上下文：多帧拼接 vs leaky integrator（30 维输入，hidden 30，6 tiles；{lr_note}）")
    print(agg.to_string(index=False))
    print("\n与 6 × 5 帧的同 seed 配对差：")
    for (kind, feat), g in runs[runs.features != "frames"].groupby(["kind", "features"]):
        ref = runs[(runs.kind == kind) & (runs.features == "frames")].set_index("seed").auc
        d = (g.set_index("seed").auc - ref).dropna()
        if len(d):
            print(f"  {kind:5s} {feat:7s} {d.mean():+.4f} ± {d.std(ddof=1):.4f}"
                  f"   逐 seed {' '.join(f'{v:+.3f}' for v in d)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
