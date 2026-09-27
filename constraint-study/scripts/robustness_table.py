"""鲁棒性表：报告的六条结论在 v1（已寄出）→ v1.1（STE 修复）→ v2（新物理模型）下的数字。

只读 results/*.csv，不做任何训练或评估；缺哪一版的数据就填 "—"。
"成立 / 不成立"的判断要结合保留意见，由人写进报告，这里只负责把数字取准。

    python scripts/robustness_table.py          # 打印并写 results/robustness.md

数据来源
    v1    noise.csv  arch_runs.csv  train_noise_cross.csv  bn_calib.csv  bits_r0.04.csv  sweep.csv
    v1.1  *_stefix*.csv（run_v11.sh、run_fig4_stefix.sh 等）
    v2    v2_runs.csv  v2_fs.csv（run_v2.sh）
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.experiment import find_nonconverged  # noqa: E402

R = Path(__file__).resolve().parent.parent / "results"
NA = "—"


def read(name: str, group_keys: list[str] | None = None) -> pd.DataFrame | None:
    """读逐 run 的 CSV；给了 ``group_keys`` 就按验证集损失剔除未收敛的 run（与扫描脚本同一规则）。"""
    f = R / name
    if not f.exists():
        return None
    df = pd.read_csv(f)
    if group_keys and "run_name" in df:
        groups = {k: sorted(g.run_name.unique()) for k, g in df.groupby(group_keys)}
        bad = find_nonconverged(R, groups, 2.0)
        if bad:
            print(f"[{name}] 剔除未收敛：{', '.join(sorted(bad))}")
        df = df[~df.run_name.isin(bad)]
    return df


def ms(mean: float, std: float | None = None, nd: int = 3) -> str:
    if mean is None or (isinstance(mean, float) and np.isnan(mean)):
        return NA
    if std is None or np.isnan(std):
        return f"{mean:.{nd}f}"
    return f"{mean:.{nd}f} ± {std:.{nd}f}"


def signed(mean: float, std: float | None = None) -> str:
    if mean is None or np.isnan(mean):
        return NA
    return f"{mean:+.3f}" + (f" ± {std:.3f}" if std is not None and not np.isnan(std) else "")


# ---------------------------------------------------------------------------
# 取数
# ---------------------------------------------------------------------------
def bits_point(df: pd.DataFrame | None, bits: int) -> str:
    if df is None:
        return NA
    r = df[(df.condition == "nominal") & (df.bits == bits)]
    return ms(r.auc_mean.iloc[0], r.auc_std.iloc[0]) if len(r) else NA


def v2_point(runs: pd.DataFrame | None, preset: str, arch: str = "none") -> str:
    """跨 (seed × 芯片) 的 mean ± std，与 aggregate() 同口径。"""
    if runs is None:
        return NA
    d = runs[(runs.preset == preset) & (runs.arch == arch)]
    return ms(d.auc.mean(), d.auc.std()) if len(d) else NA


def paired_arch_diff(runs: pd.DataFrame | None, key: str, value, sigma_col: str | None = None,
                     sigma=None) -> str:
    """none − per_layer，同 seed 配对（先对芯片取平均）。"""
    if runs is None:
        return NA
    d = runs[runs[key] == value]
    if sigma_col is not None:
        d = d[np.isclose(d[sigma_col], sigma)]
    per = d.groupby(["arch", "seed"]).auc.mean().unstack("arch")
    if not {"none", "per_layer"} <= set(per.columns):
        return NA
    diff = (per["none"] - per["per_layer"]).dropna()
    return signed(diff.mean(), diff.std(ddof=1) if len(diff) > 1 else None) if len(diff) else NA


def noise_c3_range(df: pd.DataFrame | None) -> str:
    if df is None:
        return NA
    c3 = df[(df.arm == "C3") & (df.sigma <= 0.03 + 1e-9)]
    return f"{c3.auc_mean.min():.3f}–{c3.auc_mean.max():.3f}" if len(c3) else NA


def noise_c2_worst(df: pd.DataFrame | None, sigma: float = 0.01) -> str:
    if df is None:
        return NA
    r = df[(df.arm == "C2") & np.isclose(df.sigma, sigma)]
    return f"{r.auc_min.iloc[0]:.3f}" if len(r) else NA


def cross(df: pd.DataFrame | None, train_sr: float, eval_sr: float = 0.0) -> str:
    if df is None:
        return NA
    d = df[np.isclose(df.train_sigma_read, train_sr) & np.isclose(df.eval_sigma_read, eval_sr)]
    return ms(d.auc.mean(), d.auc.std()) if len(d) else NA


def bn_gain(df: pd.DataFrame | None, arch: str, sr: float, windows: int = 20480) -> str:
    """逐片 BN 标定的收益：标定后 − 标定前（聚合均值之差）。"""
    if df is None:
        return NA
    d = df[(df.arch == arch) & np.isclose(df.sigma_read, sr)].set_index("calib_windows").auc_mean
    return signed(d[windows] - d[0]) if {0, windows} <= set(d.index) else NA


# ---------------------------------------------------------------------------
def main() -> int:
    v1_bits, v11_bits = read("bits_r0.04.csv"), read("bits_r0.04_b2-4-5-6-8-10_stefix.csv")
    v1_noise, v11_noise = read("noise.csv"), read("noise_stefix.csv")
    keys = ["arch", "sigma_read", "depth"]
    v1_arch, v11_arch = read("arch_runs.csv", keys), read("arch_stefix_runs.csv", keys)
    v1_cross, v11_cross = read("train_noise_cross.csv"), read("train_noise_cross_stefix.csv")
    v1_bn, v11_bn = read("bn_calib.csv"), read("bn_calib_stefix.csv")
    v2 = read("v2_runs.csv", ["preset", "arch"])
    sweep = read("sweep.csv")
    w, s, b, sc = (read(f"bits_r0.04_{k}_b4-5_stefix.csv") for k in ("w", "s", "b", "s"))
    sc = read("bits_r0.04_s_b4-5_stefix_c.csv")

    def arch_d2(df, sr):
        return paired_arch_diff(df[df.depth == 2] if df is not None else None, "sigma_read", sr)

    t2 = NA
    if sweep is not None:
        a = sweep.set_index("cfg_name").auc_mean
        if {"m6f5_h30b2z4", "m30f1_h30b2z4"} <= set(a.index):
            t2 = signed(a["m6f5_h30b2z4"] - a["m30f1_h30b2z4"])

    rows = [
        ("1 · 6 tiles 可行", "目标配置 none、标称噪声、6-bit",
         bits_point(v1_bits, 6), bits_point(v11_bits, 6),
         f"固定满量程 {v2_point(v2, 'fs')}；+ clipped ReLU {v2_point(v2, 'fs_crelu')}；"
         f"+ 绝对噪声 0.03 {v2_point(v2, 'fs_crelu_abs0.03')}"),
        ("2 · 时间胜过频率", "6×5 帧 − 30×1 帧（fp32）",
         t2, "fp32，不受 STE 影响", "未重跑（fp32）；leaky integrator 见 0-4"),
        ("3 · HWA 必需，σ_prog ≤ 3 % 基本无代价", "C3，σ_prog ≤ 3 % 的 AUC 范围",
         noise_c3_range(v1_noise), noise_c3_range(v11_noise), "未在 v2 下重扫 σ_prog"),
        ("", "C2（不做 HWA）σ_prog 1 % 的最差芯片",
         noise_c2_worst(v1_noise), noise_c2_worst(v11_noise), "—"),
        ("4 · 去掉层间 A/D 代价 ≤ 0.01（< 0.5 LSB）", "none − per_layer，σ_read 4 %（≈ 0.24 LSB）",
         arch_d2(v1_arch, 0.04), arch_d2(v11_arch, 0.04),
         f"固定满量程 {paired_arch_diff(v2, 'preset', 'fs')}；"
         f"+ clipped ReLU {paired_arch_diff(v2, 'preset', 'fs_crelu')}"),
        ("", "none − per_layer，σ_read 8 %（≈ 0.47 LSB）",
         arch_d2(v1_arch, 0.08), arch_d2(v11_arch, 0.08), "见子表 A（各噪声模型）"),
        ("", "none − per_layer，σ_read 17 %（≈ 1 LSB）",
         arch_d2(v1_arch, 0.17), arch_d2(v11_arch, 0.17), "见子表 B（动态范围扫描）"),
        ("5 · 训练配方决定鲁棒性", "安静芯片上：训练不加 / 加 4 % 激活噪声",
         f"{cross(v1_cross, 0.0)} / {cross(v1_cross, 0.04)}",
         f"{cross(v11_cross, 0.0)} / {cross(v11_cross, 0.04)}",
         "机制：0-2 的 fp32 对照证明是正则化"),
        ("", "逐片 BN 标定收益（配方对 / 配方错，20480 窗口）",
         f"{bn_gain(v1_bn, 'none', 0.04)} / {bn_gain(v1_bn, 'none', 0.0)}",
         f"{bn_gain(v11_bn, 'none', 0.04)} / {bn_gain(v11_bn, 'none', 0.0)}", "—"),
        ("6 · 位宽", "W/S/B 同步 4-bit",
         bits_point(v1_bits, 4), bits_point(v11_bits, 4),
         f"固定满量程 {v2_point(v2, 'fs_b4')}；+ 去直流 {v2_point(v2, 'fs_b4_c')}"),
        ("", "只降 W / S / B 到 4-bit；S=4 去直流",
         NA, " / ".join([bits_point(w, 4), bits_point(s, 4), bits_point(b, 4)])
         + f"；{bits_point(sc, 4)}", f"6-bit 去直流参照 {v2_point(v2, 'fs_c')}"),
    ]
    head = ("| 结论 | 指标 | v1（已寄出） | v1.1（STE 修复） | v2（新物理模型） |\n"
            "|---|---|---|---|---|")
    body = "\n".join(f"| {a} | {b_} | {c} | {d} | {e} |" for a, b_, c, d, e in rows)
    out = [f"## 鲁棒性表\n\n{head}\n{body}\n"]

    if v2 is not None:
        presets = ["fs", "fs_crelu", "fs_crelu_prop", "fs_crelu_shot",
                   "fs_crelu_abs0.01", "fs_crelu_abs0.03", "fs_crelu_abs0.1", "fs_crelu_abs0.3"]
        lines = ["| 预设 | none | per_layer | none − per_layer（配对） |", "|---|---|---|---|"]
        for p in presets:
            lines.append(f"| {p} | {v2_point(v2, p, 'none')} | {v2_point(v2, p, 'per_layer')} | "
                         f"{paired_arch_diff(v2, 'preset', p)} |")
        out.append("### 子表 A：各物理模型下的两种 A/D 放置\n\n" + "\n".join(lines) + "\n")

        lines = ["| 绝对噪声 σ（U 为单位） | 单列动态范围 | none | per_layer |", "|---|---|---|---|"]
        for sig in (0.01, 0.03, 0.1, 0.3):
            p = f"fs_crelu_abs{sig:g}"
            lines.append(f"| {sig:g} | {20 * np.log10(30 / sig):.0f} dB | {v2_point(v2, p, 'none')} | "
                         f"{v2_point(v2, p, 'per_layer')} |")
        lines.append(f"| tanh，0.03 | 60 dB | {v2_point(v2, 'fs_tanh_abs0.03', 'none')} | — |")
        out.append("### 子表 B：动态范围需求（固定满量程 + clipped ReLU + 绝对噪声）\n\n"
                   + "\n".join(lines) + "\n")

    fs = read("v2_fs.csv")
    if fs is not None:
        g = fs.groupby(["preset", "arch"])[["da_full_scale", "ad_full_scale", "da_clip_rate"]].agg(
            ["mean", "std"])
        lines = ["| 预设 | 架构 | D/A 满量程 | A/D 满量程 | D/A 截断率 |", "|---|---|---|---|---|"]
        for (p, a), r in g.iterrows():
            lines.append(f"| {p} | {a} | {ms(r[('da_full_scale', 'mean')], r[('da_full_scale', 'std')], 1)} | "
                         f"{ms(r[('ad_full_scale', 'mean')], r[('ad_full_scale', 'std')], 1)} | "
                         f"{r[('da_clip_rate', 'mean')]:.2%} |")
        out.append("### 子表 C：转换器满量程（log-mel dB；去直流的预设是去直流之后的量）\n\n"
                   + "\n".join(lines) + "\n")

    text = "\n".join(out)
    (R / "robustness.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
