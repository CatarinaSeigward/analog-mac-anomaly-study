"""把网页（site/index.html）要用的数字从 results/ 导出成 site/data.js。

    python scripts/export_site_data.py

★ 和 make_figures.py 一样**不做任何计算**：逐片 AUC 原样取自 *_runs.csv，汇总数取自同名汇总 CSV，
  并逐组核对两者的均值一致 —— 网页上的每个点都是一次实测，不插值。
★ 输出是 ``window.SITE_DATA = {...}`` 形式的 JS，而不是 JSON：这样网页直接双击打开（file://）也能加载。

数据来源
    headline   sweep.csv（fp32：参考模型、6-tile 模型）；bits_r0.04_b2-4-5-6-8-10_stefix.csv（6-bit 芯片）
    tiles      results/arch_none_r0.04_d2_stefix_s0/model.pt（目标配置，6-bit 权重码）
    prog       noise_stefix_runs.csv（图 2：编程误差 × 是否 HWA）
    adc        arch_stefix_runs.csv，depth = 2（图 3A：读噪声 × A/D 放置）
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.models.quant import qmax_of  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
OUT = ROOT.parent / "site" / "data.js"

CKPT = R / "arch_none_r0.04_d2_stefix_s0" / "model.pt"
LSB_PER_SIGMA_READ = 5.93   # σ_read → 噪声 / A/D LSB，只对 v1 比例噪声模型成立（报告 §3.3）


def _check_mean(runs: pd.DataFrame, agg: pd.DataFrame, keys: list[str], label: str) -> None:
    """逐组核对：由逐片数据算出的均值必须等于汇总 CSV 里的均值。"""
    got = runs.groupby(keys)["auc"].mean()
    for _, row in agg.iterrows():
        k = tuple(row[c] for c in keys)
        if abs(got.loc[k] - row["auc_mean"]) > 1e-9:
            raise SystemExit(f"{label}: {k} 的逐片均值 {got.loc[k]:.6f} ≠ 汇总 {row['auc_mean']:.6f}")


def _chips(df: pd.DataFrame) -> list[list[float]]:
    """[[seed, chip, auc], ...]，按 seed、chip 排序，网页据此在滑块切换时保持每颗芯片的身份。"""
    df = df.sort_values(["seed", "chip"])
    return [[int(s), int(c), round(float(a), 4)] for s, c, a in zip(df.seed, df.chip, df.auc)]


def headline() -> dict:
    sweep = pd.read_csv(R / "sweep.csv").set_index("cfg_name")
    ref, tgt = sweep.loc["m128f5_h128b4z8"], sweep.loc["m6f5_h30b2z4"]
    bits = pd.read_csv(R / "bits_r0.04_b2-4-5-6-8-10_stefix.csv")
    chip = bits[(bits.condition == "nominal") & (bits.bits == 6)].iloc[0]
    return {
        "ref": {"tiles": int(ref.tiles), "params": int(ref.params),
                "auc": round(float(ref.auc_mean), 3), "auc_std": round(float(ref.auc_std), 3)},
        "target": {"tiles": int(tgt.tiles), "params": int(tgt.params),
                   "auc_fp32": round(float(tgt.auc_mean), 3),
                   "auc_chip": round(float(chip.auc_mean), 3), "auc_chip_std": round(float(chip.auc_std), 3),
                   "n_seeds": int(chip.n_seeds), "n_chips": int(chip.n_chips)},
        "time_vs_freq": round(float(tgt.auc_mean - sweep.loc["m30f1_h30b2z4"].auc_mean), 3),
    }


def tiles() -> tuple[list[dict], int]:
    """6 个线性层的 6-bit 权重码（每层恰好一个 tile，所以每层一个尺度）。"""
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd, bits = ck["state_dict"], ck["analog_params"]["w_bits"]
    qmax = qmax_of(bits)
    names = ["encoder.0.0", "encoder.1.0", "encoder.2.0", "decoder.0.0", "decoder.1.0", "out"]
    out = []
    for n in names:
        w = sd[f"{n}.weight"].float()                     # (out, in)
        codes = torch.clamp(torch.round(w / (w.abs().max() / qmax)), -qmax, qmax).int()
        out.append({"out": w.shape[0], "in": w.shape[1], "codes": codes.tolist()})
    return out, bits


def prog() -> dict:
    runs = pd.read_csv(R / "noise_stefix_runs.csv")
    agg = pd.read_csv(R / "noise_stefix.csv")
    _check_mean(runs, agg, ["arm", "sigma"], "noise_stefix")
    sigmas = sorted(runs[runs.arm == "C3"].sigma.unique())
    arms = {a: {f"{s:g}": _chips(runs[(runs.arm == a) & (runs.sigma == s)]) for s in sigmas}
            for a in ("C2", "C3")}
    fp32 = agg[agg.arm == "C1"].iloc[0]
    return {"sigmas": [float(s) for s in sigmas], "fp32": round(float(fp32.auc_mean), 3),
            "normal": arms["C2"], "hwa": arms["C3"]}


def adc() -> dict:
    runs = pd.read_csv(R / "arch_stefix_runs.csv")
    agg = pd.read_csv(R / "arch_stefix.csv")
    _check_mean(runs, agg, ["depth", "sigma_read", "arch"], "arch_stefix")
    runs = runs[runs.depth == 2]
    sigmas = sorted(runs.sigma_read.unique())
    arms = {a: {f"{s:g}": _chips(runs[(runs.arch == a) & (runs.sigma_read == s)]) for s in sigmas}
            for a in ("none", "per_layer")}
    return {"sigmas": [float(s) for s in sigmas],
            "lsb": [round(s * LSB_PER_SIGMA_READ, 2) for s in sigmas],
            "none": arms["none"], "per_layer": arms["per_layer"]}


def main() -> None:
    t, w_bits = tiles()
    data = {"generated": date.today().isoformat(), "headline": headline(),
            "tiles": t, "w_bits": w_bits, "prog": prog(), "adc": adc()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("// 由 constraint-study/scripts/export_site_data.py 生成，不要手改。\n"
                   f"window.SITE_DATA = {json.dumps(data, separators=(',', ':'))};\n", encoding="utf-8")
    print(f"wrote {OUT}  ({OUT.stat().st_size / 1024:.1f} KB)")
    h = data["headline"]
    print(f"  ref {h['ref']['tiles']} tiles AUC {h['ref']['auc']}  |  target {h['target']['tiles']} tiles "
          f"fp32 {h['target']['auc_fp32']} chip {h['target']['auc_chip']}  |  time-vs-freq +{h['time_vs_freq']}")


if __name__ == "__main__":
    main()
