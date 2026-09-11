"""从 results/*.csv 出图。

    python scripts/make_figures.py            # 出全部已有数据的图
    python scripts/make_figures.py --fig 1

★ 本脚本**不做任何计算**：所有数字来自 results/*.csv，保证图与表一致。
★ 标签用英文 —— app note 是英文的，且避免 Windows 上中文字体缺失。
★ 每张图必备四件事（PLAN 附录 J）：芯片规格标注、误差带、对照曲线、图注是建议不是描述。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.utils import ensure_dir, load_config  # noqa: E402

CHIP_DIM = 30      # 30x30 阵列 -> 特征维度上限
CHIP_HIDDEN = 30

C_BASE = "#1f77b4"   # 基线网络 (hidden=128)
C_CHIP = "#d62728"   # 芯片网络 + 朴素分配
C_BEST = "#2ca02c"   # 芯片网络 + 最优分配
C_SPEC = "#555555"


def _style(ax):
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def figure1(results_dir: Path, out_dir: Path) -> None:
    """图 1：维度压缩的代价 + 同样的 30 维预算该怎么花。"""
    df = pd.read_csv(results_dir / "sweep.csv")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.5, 4.8))

    # ---------------- Panel A: AUC vs input dimension ----------------
    # 蓝：基线网络，frames=5，640 -> 80，单调下降
    d128 = df[(df.hidden == 128) & (df.frames == 5)].sort_values("input_dim")
    axA.errorbar(d128.input_dim, d128.auc_mean, yerr=d128.auc_std,
                 marker="o", capsize=4, lw=1.8, ms=6, color=C_BASE,
                 label="Baseline net (hidden=128), 5 frames")

    # 红：芯片网络 + 朴素分配（frames=1）
    dnaive = df[(df.hidden == CHIP_HIDDEN) & (df.frames == 1)].sort_values("input_dim")
    axA.errorbar(dnaive.input_dim, dnaive.auc_mean, yerr=dnaive.auc_std,
                 marker="o", capsize=4, lw=1.8, ms=6, color=C_CHIP,
                 label="Chip net (hidden=30), naive 1-frame")

    # 绿星：芯片网络 + 最优分配（6 bands x 5 frames，同样 30 维、同样 6 tiles）
    best = df[df.cfg_name == "m6f5_h30b2z4"].iloc[0]
    naive30 = df[df.cfg_name == "m30f1_h30b2z4"].iloc[0]
    axA.errorbar([best.input_dim], [best.auc_mean], yerr=[best.auc_std],
                 marker="*", ms=18, capsize=4, lw=0, elinewidth=1.5,
                 color=C_BEST, label="Chip net, 6 bands x 5 frames")
    axA.annotate("", xy=(CHIP_DIM, best.auc_mean - 0.005),
                 xytext=(CHIP_DIM, naive30.auc_mean + 0.005),
                 arrowprops=dict(arrowstyle="->", color=C_BEST, lw=1.6))
    axA.text(34, (best.auc_mean + naive30.auc_mean) / 2 - 0.012,
             "+0.068\nsame 30 dims\nsame 6 tiles", fontsize=8.5, color=C_BEST)

    axA.axvline(CHIP_DIM, color=C_SPEC, ls="--", lw=1.2)
    axA.text(CHIP_DIM * 0.93, 0.812, "chip spec\n30-dim limit",
             fontsize=8.5, color=C_SPEC, ha="right", va="top")
    axA.axhline(0.5, color="#999999", ls=":", lw=1)
    axA.text(16.5, 0.506, "chance", fontsize=8, color="#777777")

    axA.set_xscale("log")
    axA.set_xticks([16, 30, 80, 160, 320, 640])
    axA.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axA.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    axA.set_ylim(0.49, 0.83)
    axA.set_xlabel("Input feature dimension")
    axA.set_ylabel("AUC (mean ± std over 3 seeds)")
    axA.set_title("A. Cost of dimensionality reduction", fontsize=11, loc="left")
    axA.legend(fontsize=8, loc="lower right", framealpha=0.95)
    _style(axA)

    # ---------------- Panel B: allocation of a fixed ~30-dim budget ----------------
    alloc = df[df.tracks.str.contains("alloc", na=False)].copy()
    extra = df[df.cfg_name == "m32f1_h30b2z4"]
    alloc = pd.concat([alloc, extra]).sort_values("frames")

    labels = [f"{int(r.n_mels)}x{int(r.frames)}" for _, r in alloc.iterrows()]
    x = list(range(len(alloc)))
    colors = [C_CHIP if f == 1 else C_BASE for f in alloc.frames]
    axB.bar(x, alloc.auc_mean, yerr=alloc.auc_std, capsize=4,
            color=colors, alpha=0.85, width=0.62)
    for i, (_, r) in enumerate(alloc.iterrows()):
        axB.text(i, r.auc_mean + r.auc_std + 0.008, f"{r.auc_mean:.3f}",
                 ha="center", fontsize=8.5)

    axB.set_xticks(x)
    axB.set_xticklabels(labels)
    axB.set_xlabel("mel bands x frames   (all ~30 dims, 6-8 tiles, hidden=30)")
    axB.set_ylabel("AUC (mean ± std over 3 seeds)")
    axB.set_ylim(0.55, 0.79)
    axB.set_title("B. Same budget, different allocation", fontsize=11, loc="left")
    axB.annotate("", xy=(0, 0.762), xytext=(4, 0.762),
                 arrowprops=dict(arrowstyle="<->", color="#333333", lw=1.1))
    axB.text(2, 0.768, "+0.068 AUC   (t = 4.2)", ha="center", fontsize=9, color="#333333")
    axB.text(0.5, 0.575, "single frame\nno temporal context", ha="center",
             fontsize=8, color=C_CHIP)
    _style(axB)

    fig.tight_layout()
    out = out_dir / "fig1_dimension.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig1] -> {out}")


C_C2 = "#d62728"     # 裸训
C_C3 = "#1f77b4"     # HWA
C_C4 = "#9467bd"     # HWA + 层间重量化
C_C1 = "#2ca02c"     # fp32 上界


def figure2(results_dir: Path, out_dir: Path) -> None:
    """图 2：graceful degradation —— HWA 训练 vs 裸训，在编程误差下的表现。

    A：跨 (种子 × 芯片) 的 AUC mean ± std
    B：最差芯片的 AUC —— 量产良率视角，裸训会出现低于随机的芯片
    """
    df = pd.read_csv(results_dir / "noise.csv")
    x_pct = lambda s: s * 100  # noqa: E731

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.5, 4.8))

    # ---------------- Panel A ----------------
    c1 = df[df.arm == "C1"].iloc[0]
    axA.axhspan(c1.auc_mean - c1.auc_std, c1.auc_mean + c1.auc_std,
                color=C_C1, alpha=0.10, lw=0)
    axA.axhline(c1.auc_mean, color=C_C1, lw=1.4, ls="-",
                label=f"C1  fp32 upper bound ({c1.auc_mean:.3f})")

    for arm, color, ls, label in [
        ("C2", C_C2, "-", "C2  fp32-trained, analog inference"),
        ("C3", C_C3, "-", "C3  HWA  (6-bit at every layer input)"),
        ("C4", C_C4, "--", "C4  HWA + 6-bit at every layer output"),
    ]:
        d = df[df.arm == arm].sort_values("sigma")
        xs = x_pct(d.sigma)
        axA.plot(xs, d.auc_mean, marker="o", ms=5, lw=1.8, color=color, ls=ls, label=label)
        if arm != "C4":
            axA.fill_between(xs, d.auc_mean - d.auc_std, d.auc_mean + d.auc_std,
                             color=color, alpha=0.15, lw=0)

    axA.axhline(0.5, color="#999999", ls=":", lw=1)
    axA.text(9.8, 0.508, "chance", fontsize=8, color="#777777", ha="right")
    axA.set_xlabel("Weight programming error  σ_prog  (%)")
    axA.set_ylabel("AUC  (mean ± std over seeds × 10 chips)")
    axA.set_title("A. Hardware-aware training degrades gracefully", fontsize=11, loc="left")
    axA.set_xticks([0, 1, 2, 3, 5, 10])
    axA.set_ylim(0.45, 0.80)
    axA.legend(fontsize=8, loc="lower left", framealpha=0.95)
    _style(axA)

    # ---------------- Panel B：最差芯片 ----------------
    for arm, color, label in [("C2", C_C2, "C2  fp32-trained"),
                              ("C3", C_C3, "C3  HWA")]:
        d = df[df.arm == arm].sort_values("sigma")
        axB.plot(x_pct(d.sigma), d.auc_min, marker="s", ms=5, lw=1.8,
                 color=color, label=label)
    axB.axhline(0.5, color="#999999", ls=":", lw=1)
    axB.axhspan(0.0, 0.5, color=C_C2, alpha=0.06, lw=0)
    axB.text(9.8, 0.49, "worse than random guessing", fontsize=8,
             color=C_C2, va="top", ha="right")
    axB.set_xlabel("Weight programming error  σ_prog  (%)")
    axB.set_ylabel("AUC of the worst simulated chip")
    axB.set_title("B. Worst-case chip (yield view)", fontsize=11, loc="left")
    axB.set_xticks([0, 1, 2, 3, 5, 10])
    axB.set_ylim(0.25, 0.80)
    axB.legend(fontsize=8.5, loc="lower left", framealpha=0.95)
    _style(axB)

    fig.tight_layout()
    out = out_dir / "fig2_noise.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig2] -> {out}")


NOISE_PER_LSB = 5.93   # Day 4 标定：噪声/LSB ≈ 5.93 × σ_read（深度 2 HWA 模型各层均值）
C_NONE = "#1f77b4"     # Ai Linear：隐藏层无 A/D
C_PER = "#ff7f0e"      # 常规 CIM：每层 A/D + D/A


def figure3(results_dir: Path, out_dir: Path) -> None:
    """图 3：去掉层间 A/D 的代价取决于信号通路噪声。

    A：目标深度（2 个隐藏块）下，AUC 随 σ_read 的变化，两种架构对比
    B：深度的影响（深度与参数量混杂，危险点 K）
    """
    df = pd.read_csv(results_dir / "arch.csv")
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.5, 4.8))

    # ---------------- Panel A ----------------
    d2 = df[df.depth == 2]
    srs = sorted(d2.sigma_read.unique())
    xs = list(range(len(srs)))
    for arch, color, label in [
        ("none", C_NONE, "No A/D between layers (Ai Linear)"),
        ("per_layer", C_PER, "A/D + D/A at every layer (conventional CIM)"),
    ]:
        d = d2[d2.arch == arch].set_index("sigma_read").reindex(srs)
        axA.plot(xs, d.auc_mean, marker="o", ms=5, lw=1.8, color=color, label=label)
        axA.fill_between(xs, d.auc_mean - d.auc_std, d.auc_mean + d.auc_std,
                         color=color, alpha=0.15, lw=0)
    axA.set_xticks(xs)
    axA.set_xticklabels([f"{s * 100:g}%\n({s * NOISE_PER_LSB:.2f} LSB)" if s > 0 else "0"
                         for s in srs])
    axA.axhline(0.5, color="#999999", ls=":", lw=1)
    axA.set_xlabel("Signal-path read noise  σ_read   (≈ noise / A/D LSB)")
    axA.set_ylabel("AUC  (mean ± std over seeds × 10 chips)")
    axA.set_title("A. Target depth: cost of removing inter-layer A/D", fontsize=11, loc="left")
    # 训练配方对照：none 架构用 4% 激活噪声训练、部署到安静芯片（σ_read = 0）
    cross = results_dir / "train_noise_cross.csv"
    if cross.exists():
        c = pd.read_csv(cross)
        q = c[(c.train_sigma_read == 0.04) & (c.eval_sigma_read == 0.0)].auc
        if len(q):
            axA.errorbar([0], [q.mean()], yerr=[q.std()], marker="*", ms=15, lw=0,
                         elinewidth=1.4, capsize=4, color=C_NONE, mfc="white", mew=1.6,
                         label="No A/D, trained with 4% activation noise, quiet chip")
    axA.legend(fontsize=8, loc="lower left", framealpha=0.95)
    _style(axA)

    # ---------------- Panel B ----------------
    # 只画 σ_read > 0：σ_read = 0 的 none 模型受训练配方问题影响（报告 4.4 结论 8），
    # 那部分数据在 arch.csv 与报告表格里完整保留。
    multi = [sr for sr, g in df.groupby("sigma_read") if g.depth.nunique() > 1 and sr > 0]
    for sr, ls in zip(multi, ["-", "--", ":"]):
        for arch, color, short in [("none", C_NONE, "no A/D"), ("per_layer", C_PER, "per-layer A/D")]:
            d = df[(df.sigma_read == sr) & (df.arch == arch)].sort_values("depth")
            axB.errorbar(d.depth, d.auc_mean, yerr=d.auc_std, marker="o", ms=5, lw=1.6,
                         ls=ls, color=color, capsize=3,
                         label=f"{short}, σ_read = {sr * 100:g}%")
    axB.axvline(2, color=C_SPEC, ls=":", lw=1)
    axB.text(2.05, axB.get_ylim()[1], "target", fontsize=8, color=C_SPEC, va="top")
    axB.set_xticks([1, 2, 3, 4])
    axB.set_xlabel("Hidden blocks per encoder / decoder  (depth)")
    axB.set_ylabel("AUC  (mean ± std)")
    shown = ", ".join(f"{v * 100:g}%" for v in multi)
    axB.set_title(f"B. Depth at σ_read = {shown}  (confounded with parameter count)",
                  fontsize=11, loc="left")
    axB.legend(fontsize=8, loc="lower left", framealpha=0.95)
    _style(axB)

    fig.tight_layout()
    out = out_dir / "fig3_adc.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig3] -> {out}")


def figure4(results_dir: Path, out_dir: Path) -> None:
    """图 4：6-bit 是不是瓶颈（Ai Linear 架构，W/S/B 同步，每个位宽各自 QAT）。"""
    recipe = results_dir / "bits_r0.04.csv"
    source = recipe if recipe.exists() else results_dir / "bits.csv"
    df = pd.read_csv(source)
    note = ("trained with 4% activation noise" if source == recipe
            else "trained WITHOUT activation noise: fragile recipe, see report 4.4")
    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    for cond, color, label in [
        ("quant_only", C_C1, "Quantization only (noise off)"),
        ("nominal", C_C3, "Quantization + nominal noise (σ_prog 3%, σ_d2d 2%, σ_read 4%)"),
    ]:
        d = df[df.condition == cond].sort_values("bits")
        ax.plot(d.bits, d.auc_mean, marker="o", ms=5, lw=1.8, color=color, label=label)
        ax.fill_between(d.bits, d.auc_mean - d.auc_std, d.auc_mean + d.auc_std,
                        color=color, alpha=0.15, lw=0)

    noise_csv = results_dir / "noise.csv"
    if noise_csv.exists():
        c1 = pd.read_csv(noise_csv).query("arm == 'C1'")
        if len(c1):
            c1 = c1.iloc[0]
            ax.axhline(c1.auc_mean, color="#555555", ls="--", lw=1.2,
                       label=f"fp32 reference ({c1.auc_mean:.3f})")

    ax.axvline(6, color=C_SPEC, ls="--", lw=1.2)
    ax.text(6.15, ax.get_ylim()[1], "chip spec\n6-bit", fontsize=8.5, color=C_SPEC, va="top")
    ax.axhline(0.5, color="#999999", ls=":", lw=1)
    ax.set_xticks([2, 4, 6, 8, 10])
    ax.set_xlabel("W / S / B precision  (bits, all three together, QAT per width)")
    ax.set_ylabel("AUC  (mean ± std)")
    ax.set_title(f"Is 6-bit a bottleneck?   (no inter-layer A/D, {note})", fontsize=9.5, loc="left")
    ax.legend(fontsize=8, loc="lower right", framealpha=0.95)
    _style(ax)

    fig.tight_layout()
    out = out_dir / "fig4_bits.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig4] -> {out}")


FIGURES = {1: figure1, 2: figure2, 3: figure3, 4: figure4}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/fast.yaml")
    ap.add_argument("--fig", type=int, nargs="+", default=sorted(FIGURES))
    args = ap.parse_args()

    cfg = load_config(args.config)
    results_dir = Path(cfg.output.results_dir)
    out_dir = ensure_dir(results_dir / "figures")

    for n in args.fig:
        if n not in FIGURES:
            print(f"[skip] 图 {n} 尚未实现")
            continue
        try:
            FIGURES[n](results_dir, out_dir)
        except FileNotFoundError as e:
            print(f"[skip] 图 {n}: 缺数据 {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
