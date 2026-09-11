"""[已审阅 · 暂缓] Day 4 子实验：多 tile 权衡 —— 阵列该不该做大？

============================================================================
★ 审稿结论（Day 4）：暂缓运行，等硬件团队给出读出噪声规格。

  下方原草稿第 1 条的论证是对的：逐元素独立的乘性权重噪声与分块方式无关。
  这恰恰说明，在当前噪声模型下"多 tile"**没有任何额外代价**，实验只会平凡地
  得出"越大越好"。多 tile 的真实代价只可能来自**每个 tile 部分和的读出噪声**，
  而这个代价是否存在，完全取决于读出噪声的性质：
    - 绝对的（每列一个固定噪声底，与信号无关） -> tile 越多噪声方差线性累加，有代价
    - 相对的（与信号电流成比例）               -> 部分和的信噪比不随拆分变化，无代价
  没有器件数据时，结论等于建模选择本身，不是实验发现。已列入 assumptions.md
  的待确认清单（第 8 条）。

  以下为原草稿的检查清单，保留作记录。
============================================================================
⚠️  （原草稿）本文件是 Day 2 结束时提前写的草稿，**尚未运行、尚未审阅**。
    进行到 Day 4 时先确认：

    1. ★★ ROADMAP 危险点 L：**噪声必须逐 tile 注入**。
       如果整个权重矩阵只注入一次噪声，"更多 tile = 更多噪声源"这个
       核心权衡就完全消失了，本实验直接失去意义。
       当前 src/models/analog.py 的 AnalogLinear 是对整个权重矩阵注入噪声 ——
       在 60 维/2 tile 的场景下，这**恰好等价于**逐 tile 独立注入
       （因为噪声是逐元素独立的乘性噪声，与分块方式无关）。
       ⚠️ **但如果后续改成 per-tile 的相关噪声（如共享一个 tile 级偏置），
          就必须重新检查这一点。**
    2. 假设 A7：**芯片是否支持多 tile 级联尚未确认**。若不支持，
       本实验只有理论意义，须在 app note 里标明。
    3. 输入维度方向的分块在模拟域是电流求和（KCL），电路上很便宜；
       输出维度方向的分块几乎无代价。真正的代价是**参与器件变多 -> 累积噪声**。
============================================================================

研究问题
    在 30x30 阵列上，用 2 个 tile 承载 60 维输入，比用 1 个 tile 承载 30 维，
    净收益是正还是负？

      更多输入维度  -> 更多信息（Day 2 已证实：维度与 AUC 正相关）
      更多 tile     -> 更多器件参与 -> 累积噪声更大

    这个问题直接回答"阵列该不该做大"，是 app note 第 5 节最有价值的内容之一。

设计
    固定分配策略为最优的时频比（Day 2 结论：多帧优先），比较：

      | 配置          | 输入维度 | 输入方向 tile 数 |
      |---------------|---------|-----------------|
      | 6 x 5         |   30    |        1        |
      | 12 x 5        |   60    |        2        |
      | 18 x 5        |   90    |        3        |
      | 24 x 5        |  120    |        4        |

    在 σ = 0（无噪声）与 σ = 标称 两种条件下各跑一遍：
      - σ=0 时应单调上升（纯信息增益）
      - σ>0 时若出现拐点，那个拐点就是"阵列做多大最划算"的答案

    ★ 需要新建 n_mels ∈ {12, 18, 24} 的缓存：
        python scripts/prepare_data.py --n-mels 12 18 24

产出
    results/multitile_runs.csv / results/multitile.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src.noise import DeviceParams  # noqa: E402
from src.tiling import layer_tiles  # noqa: E402
from src.utils import load_config  # noqa: E402

FRAMES = 5                       # Day 2 结论：帧数优先
MEL_BANDS = [6, 12, 18, 24]      # -> 30 / 60 / 90 / 120 维
HIDDEN = 30
SIGMAS = {"noise-free": 0.0, "nominal": 0.03}
SEEDS = [0, 1, 2]
N_CHIPS = 10
TILE = 30


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/fast.yaml")
    ap.add_argument("--mel-bands", type=int, nargs="+", default=MEL_BANDS)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--n-chips", type=int, default=N_CHIPS)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    base = load_config(args.config)
    results_dir = Path(base.output.results_dir)

    print(f"{'n_mels':>8}{'dim':>6}{'in-tiles':>10}{'layer1 tiles':>14}")
    for nm in args.mel_bands:
        dim = nm * FRAMES
        in_tiles = -(-dim // TILE)               # ceil
        print(f"{nm:>8}{dim:>6}{in_tiles:>10}{layer_tiles(HIDDEN, dim, TILE):>14}")

    n = len(args.mel_bands) * len(args.seeds)
    print(f"\n计划：{n} 次训练，{n * len(SIGMAS) * args.n_chips} 次评估")
    if args.dry_run:
        return 0

    rows = []
    for nm in args.mel_bands:
        for seed in args.seeds:
            # TODO(Day 4): 接 Day 3 的 train_hwa()
            #   cfg = merge(base, name=f"mt{nm}_s{seed}", feature.n_mels=nm, ...)
            #   model = train_hwa(cfg, DeviceParams(sigma_prog=SIGMA_TRAIN))
            for gname, sigma in SIGMAS.items():
                # params = DeviceParams(sigma_prog=sigma)
                # df = evaluate_chips(model, ..., n_chips=args.n_chips, params=params)
                # rows += [{"n_mels": nm, "input_dim": nm*FRAMES, "seed": seed,
                #           "sigma_group": gname, "sigma": sigma, **r}
                #          for r in df.to_dict("records")]
                raise NotImplementedError(
                    "草稿：需要接 Day 3 的 train_hwa() 与 evaluate_chips()"
                )

    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "multitile_runs.csv", index=False)
    agg = (df.groupby(["input_dim", "sigma_group"])
             .agg(n=("auc", "size"), auc_mean=("auc", "mean"), auc_std=("auc", "std"))
             .reset_index())
    agg.to_csv(results_dir / "multitile.csv", index=False)
    print(agg.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
