"""Day 3：噪声扫描 —— C1/C2/C3/C4 四组对照 → 图 2（全研究最重要的一张图）。

    python scripts/sweep_noise.py --dry-run
    python scripts/sweep_noise.py --seeds 0 1 2 --n-chips 10

四组对照（缺一组，对应的结论就站不住）::

    C1  fp32 训练 + fp32 推理            压缩后的无噪声上界
    C2  fp32 训练 + 6-bit/噪声 推理      ★ 反面对照：不做 HWA 会怎样
    C3  HWA 训练 + 6-bit/噪声 推理       ★ 主方案：graceful degradation
    C4  同 C3 但层间重量化               Day 4 原创点的对照（此处顺带产出）

关键设计（ROADMAP §2.1）
    - **D2D 是评估期变量**：训练一次，部署到 N 颗仿真芯片分别评估。
    - C2 复用 C1 的 fp32 checkpoint（不重新训练）—— 它们本来就是同一个模型，
      只是推理方式不同。
    - C4 复用 C3 的 checkpoint，只改 ``requantize_output`` —— 保证两者的
      训练随机性与芯片采样完全一致（危险点 H）。
    - C3 采用 **σ_train = σ_eval**（匹配训练）。σ_train/σ_eval 比值的单独
      扫描留给后续（PLAN 附录 E.3）。

x 轴定义
    扫 ``sigma_prog``（权重编程误差），``sigma_d2d`` 固定在标称值 —— 这样图 2
    的横轴含义明确："program-verify 容差带对应的权重误差"。
    注意 σ_prog=0 处仍有 D2D 噪声，这是真实情况。

产出
    results/noise_runs.csv  每 (arm, sigma, seed, chip) 一行
    results/noise.csv       按 (arm, sigma) 聚合 mean ± std
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json  # noqa: E402
import statistics  # noqa: E402

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from src.evaluate_analog import evaluate_chips  # noqa: E402
from src.features.logmel import cache_prefix, load_cache  # noqa: E402
from src.models.analog import analogize, count_tiles  # noqa: E402
from src.models.autoencoder import build_from_config  # noqa: E402
from src.noise import IDEAL, DeviceParams  # noqa: E402
from src.train import train  # noqa: E402
from src.utils import get_device, load_config  # noqa: E402

# Day 2 选定的目标配置：6 mel 带 x 5 帧 = 30 维，hidden=30，6 个 tile
TARGET = {"n_mels": 6, "frames": 5, "hidden": 30, "n_blocks": 2, "bottleneck": 4}

SIGMAS = [0.0, 0.01, 0.02, 0.03, 0.05, 0.10]   # 权重编程误差
SIGMA_D2D = 0.02                                # 器件间失配，固定在标称值
SEEDS = [0, 1, 2]
N_CHIPS = 10


def make_params(sigma_prog: float, requantize: bool = False) -> DeviceParams:
    return DeviceParams(sigma_prog=sigma_prog, sigma_d2d=SIGMA_D2D,
                        requantize_output=requantize)


def target_cfg(base, name: str, seed: int):
    return OmegaConf.merge(base, OmegaConf.from_dotlist([
        f"name={name}", f"seed={seed}",
        f"feature.n_mels={TARGET['n_mels']}", f"feature.frames={TARGET['frames']}",
        f"model.hidden={TARGET['hidden']}", f"model.n_blocks={TARGET['n_blocks']}",
        f"model.bottleneck={TARGET['bottleneck']}",
    ]))


def load_model(cfg, results_dir: Path, name: str, device, params=None):
    """从 checkpoint 重建模型；params 非 None 时换成 AnalogLinear。"""
    ckpt = torch.load(results_dir / name / "model.pt", map_location=device,
                      weights_only=False)
    model = build_from_config(cfg, input_dim=ckpt["input_dim"])
    if params is not None:
        analogize(model, params)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


def final_val_loss(results_dir: Path, name: str) -> float | None:
    f = results_dir / name / "history.json"
    if not f.exists():
        return None
    return json.loads(f.read_text(encoding="utf-8"))["history"][-1]["val_loss"]


def find_nonconverged(results_dir: Path, names: list[str], factor: float) -> dict[str, float]:
    """按**验证损失**识别未收敛的训练。

    ★ 为什么用 val loss 而不是 AUC：
      val loss 是训练期诊断量，剔除它不涉及测试集 —— 这是合法的收敛性检查。
      按测试 AUC 剔除就变成挑数据了（PLAN 第 9 节诚信规范）。

    判据：final val loss > factor × 全体中位数，统一应用于所有 HWA 训练。
    实测中该判据把唯一一次训练塌陷（2.62× 中位数）与合理的高噪声训练
    （σ=0.10 时 1.56–1.61×）干净地分开了。
    """
    losses = {n: v for n in names if (v := final_val_loss(results_dir, n)) is not None}
    if len(losses) < 3:
        return {}
    med = statistics.median(losses.values())
    return {n: v / med for n, v in losses.items() if v > factor * med}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/fast.yaml")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--sigmas", type=float, nargs="+", default=SIGMAS)
    ap.add_argument("--n-chips", type=int, default=N_CHIPS)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--converge-factor", type=float, default=2.0,
                    help="final val loss 超过中位数这个倍数即判为未收敛并剔除")
    args = ap.parse_args()

    base = load_config(args.config)
    results_dir = Path(base.output.results_dir)
    device = get_device()

    n_fp32 = len(args.seeds)
    n_hwa = len(args.sigmas) * len(args.seeds)
    n_eval = (n_fp32                                    # C1
              + len(args.sigmas) * len(args.seeds) * args.n_chips   # C2
              + n_hwa * args.n_chips * 2)                           # C3 + C4
    print(f"目标配置 {TARGET}  ->  {count_tiles(analogize(build_from_config(target_cfg(base,'x',0), 30), IDEAL))} tiles")
    print(f"sigma_prog = {args.sigmas}   sigma_d2d = {SIGMA_D2D}（固定）")
    print(f"训练：{n_fp32} 次 fp32（C1/C2 共用）+ {n_hwa} 次 HWA（C3/C4 共用）= {n_fp32+n_hwa} 次")
    print(f"评估：{n_eval} 次，跨 {args.n_chips} 颗仿真芯片")
    if args.dry_run:
        return 0

    mel, meta = load_cache(cache_prefix(base.data.cache_dir, TARGET["n_mels"], "test"))
    frames = TARGET["frames"]
    mids = list(base.eval.machine_ids)
    max_fpr = float(base.eval.max_fpr)
    rows = []
    hwa_names: list[str] = []
    t_start = time.time()

    # ---------------- C1 / C2：fp32 训练 ----------------
    for seed in args.seeds:
        name = f"noise_fp32_s{seed}"
        if args.force or not (results_dir / name / "model.pt").exists():
            print(f"\n=== 训练 fp32 (C1/C2 共用)  seed={seed} ===")
            train(target_cfg(base, name, seed))

        cfg = target_cfg(base, name, seed)

        # C1：fp32 推理（无量化无噪声，1 颗芯片即可，D2D=0）
        m = load_model(cfg, results_dir, name, device, IDEAL)
        df = evaluate_chips(m, mel, meta, frames, device, IDEAL,
                            n_chips=1, machine_ids=mids, max_fpr=max_fpr)
        rows += [{"arm": "C1", "sigma": 0.0, "seed": seed, **r} for r in df.to_dict("records")]

        # C2：同一个 fp32 模型，换成模拟推理
        for sig in args.sigmas:
            p = make_params(sig)
            m = load_model(cfg, results_dir, name, device, p)
            df = evaluate_chips(m, mel, meta, frames, device, p,
                                n_chips=args.n_chips, machine_ids=mids, max_fpr=max_fpr)
            rows += [{"arm": "C2", "sigma": sig, "seed": seed, **r} for r in df.to_dict("records")]
        print(f"[C1/C2] seed={seed} 完成  ({time.time()-t_start:.0f}s)")

    # ---------------- C3 / C4：HWA 训练（sigma_train = sigma_eval） ----------------
    for sig in args.sigmas:
        for seed in args.seeds:
            name = f"noise_hwa_s{sig:g}_seed{seed}"
            hwa_names.append(name)
            p_train = make_params(sig)
            if args.force or not (results_dir / name / "model.pt").exists():
                print(f"\n=== HWA 训练  sigma={sig}  seed={seed} ===")
                train(target_cfg(base, name, seed), params=p_train)

            cfg = target_cfg(base, name, seed)
            for arm, requant in (("C3", False), ("C4", True)):
                p = make_params(sig, requantize=requant)
                m = load_model(cfg, results_dir, name, device, p)
                df = evaluate_chips(m, mel, meta, frames, device, p,
                                    n_chips=args.n_chips, machine_ids=mids, max_fpr=max_fpr)
                rows += [{"arm": arm, "sigma": sig, "seed": seed,
                          "run_name": name, **r} for r in df.to_dict("records")]
            print(f"[C3/C4] sigma={sig} seed={seed} 完成  ({time.time()-t_start:.0f}s)")

    # ---------------- 聚合 ----------------
    runs = pd.DataFrame(rows)
    runs.to_csv(results_dir / "noise_runs.csv", index=False)

    # ---- 收敛性过滤（只作用于 HWA 训练出来的 C3/C4）----
    bad = find_nonconverged(results_dir, sorted(set(hwa_names)), args.converge_factor)
    if bad:
        print()
        print(f"[收敛性检查] {len(bad)}/{len(set(hwa_names))} 次 HWA 训练未收敛，"
              f"判据 final val loss > {args.converge_factor}x 中位数：")
        for n, ratio in sorted(bad.items(), key=lambda x: -x[1]):
            print(f"    {n:<28} {ratio:.2f}x 中位数  -> 从聚合中剔除")
        print("    （按验证损失剔除，不涉及测试集；见 find_nonconverged 的 docstring）")
        runs = runs[~runs.get("run_name", pd.Series(dtype=str)).isin(bad)]
    else:
        print()
        print(f"[收敛性检查] 全部 {len(set(hwa_names))} 次 HWA 训练均收敛")
    agg = (runs.groupby(["arm", "sigma"])
                .agg(n=("auc", "size"),
                     n_seeds=("seed", "nunique"), n_chips=("chip", "nunique"),
                     auc_mean=("auc", "mean"), auc_std=("auc", "std"),
                     auc_min=("auc", "min"),
                     pauc_mean=("pauc", "mean"), pauc_std=("pauc", "std"))
                .reset_index())
    agg[["auc_std", "pauc_std"]] = agg[["auc_std", "pauc_std"]].fillna(0.0)
    agg.to_csv(results_dir / "noise.csv", index=False)

    print(f"\n{'='*96}\n结果 -> {results_dir/'noise.csv'}   (总耗时 {(time.time()-t_start)/60:.1f} 分钟)\n{'='*96}")
    show = agg.copy()
    show["AUC"] = show.apply(lambda r: f"{r.auc_mean:.4f} ± {r.auc_std:.4f}", axis=1)
    show["pAUC"] = show.apply(lambda r: f"{r.pauc_mean:.4f} ± {r.pauc_std:.4f}", axis=1)
    print(show[["arm", "sigma", "n_seeds", "n_chips", "n", "AUC", "pAUC", "auc_min"]]
          .to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
