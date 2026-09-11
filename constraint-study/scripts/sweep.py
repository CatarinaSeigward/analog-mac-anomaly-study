"""Day 2 维度扫描：跑一组配置 x 多个随机种子，聚合成 mean ± std。

    python scripts/sweep.py --dry-run                    # 只列计划
    python scripts/sweep.py --seeds 0 1 2                # 三个种子
    python scripts/sweep.py --track alloc chip --seeds 0 1 2
    python scripts/sweep.py --force                      # 重跑已完成的

四条 track，分别隔离不同的损失来源（见 PLAN 附录 B.4）：

  dim   只压输入维度，网络固定在基线 128/4/8   -> 隔离"特征信息损失"
  net   只压网络，输入固定 640 维              -> 隔离"网络容量损失"
  chip  芯片可行配置：输入 <= 30 且 hidden = 30 -> 两者叠加的真实代价
  alloc 固定约 30 维预算下的时频分配            -> 频率分辨率 vs 时间上下文

★ 为什么必须跑多种子（PLAN 附录 H）
  单种子结果的误差棒未知。首轮单种子扫描里出现过 16 维 (0.695) 反而优于
  30 维 (0.630) 的反常排序 —— 信息论上说不通，说明种子噪声可能有 0.05 量级，
  而不少结论的差距就在这个量级。在单种子的点上画图 = 把噪声当信号画出去。

  产出两个 CSV：
    results/sweep_runs.csv  每次运行一行（含 seed）
    results/sweep.csv       按配置聚合，带 auc_mean / auc_std / n_seeds
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from src.evaluate import evaluate  # noqa: E402
from src.features.logmel import cache_prefix  # noqa: E402
from src.tiling import max_layer_tiles, model_tiles  # noqa: E402
from src.train import train  # noqa: E402
from src.utils import load_config  # noqa: E402

BASE_NET = (128, 4, 8)     # hidden, n_blocks, bottleneck —— 官方基线
CHIP_NET = (30, 2, 4)      # 芯片可行的网络：宽度 <= 30

# (n_mels, frames, hidden, n_blocks, bottleneck)
TRACKS: dict[str, list[tuple[int, int, int, int, int]]] = {
    "dim": [
        (128, 5, *BASE_NET),   # 640
        (64, 5, *BASE_NET),    # 320
        (32, 5, *BASE_NET),    # 160
        (16, 5, *BASE_NET),    # 80
        (32, 2, *BASE_NET),    # 64
        (30, 1, *BASE_NET),    # 30
        (16, 1, *BASE_NET),    # 16
    ],
    "net": [
        (128, 5, 64, 4, 8),    # 640 输入，网络减半
        (128, 5, *CHIP_NET),   # 640 输入，网络压到芯片尺寸
    ],
    "chip": [
        (32, 1, *CHIP_NET),    # 32
        (30, 1, *CHIP_NET),    # 30
        (16, 2, *CHIP_NET),    # 32
        (16, 1, *CHIP_NET),    # 16
    ],
    "alloc": [                 # 约 30 维预算，频率 vs 时间的分配
        (30, 1, *CHIP_NET),    # 30x1  纯频率分辨率
        (16, 2, *CHIP_NET),    # 16x2
        (10, 3, *CHIP_NET),    # 10x3
        (6, 5, *CHIP_NET),     # 6x5   最大时间上下文
    ],
}

BASELINE_CFG = (128, 5, *BASE_NET)
LEGACY_BASELINE = "c0_fast80"   # Day 2 首轮的基线 run，等价于 BASELINE_CFG 的 seed 0

CFG_KEYS = ("n_mels", "frames", "hidden", "n_blocks", "bottleneck")


def cfg_name(c: tuple[int, int, int, int, int]) -> str:
    n_mels, frames, h, b, z = c
    return f"m{n_mels}f{frames}_h{h}b{b}z{z}"


def run_name(c: tuple[int, int, int, int, int], seed: int) -> str:
    return f"{cfg_name(c)}_s{seed}"


def cache_exists(cfg, n_mels: int) -> bool:
    return all(
        Path(f"{cache_prefix(cfg.data.cache_dir, n_mels, split)}_logmel.npy").exists()
        for split in ("train", "test")
    )


def migrate_legacy(results_dir: Path) -> int:
    """把首轮无 seed 后缀的 run 目录迁成 ``_s0``，避免 45 分钟的算力白跑。

    ``c0_fast80`` 是 README 里引用的锚点，保留原名并复制一份为 seed 0。
    """
    moved = 0
    for cfgs in TRACKS.values():
        for c in cfgs:
            old = results_dir / cfg_name(c)
            new = results_dir / run_name(c, 0)
            if old.exists() and not new.exists():
                shutil.move(str(old), str(new))
                print(f"  迁移 {old.name} -> {new.name}")
                moved += 1

    legacy = results_dir / LEGACY_BASELINE
    target = results_dir / run_name(BASELINE_CFG, 0)
    if legacy.exists() and not target.exists():
        shutil.copytree(legacy, target)
        print(f"  复制 {LEGACY_BASELINE} -> {target.name}（保留锚点原名）")
        moved += 1
    return moved


def build_plan(base_cfg, tracks: list[str], seeds: list[int]):
    """返回 (可跑, 缺缓存)。同一配置出现在多个 track 时合并 track 标签。"""
    merged: dict[tuple, set[str]] = {}
    for t in tracks:
        for c in TRACKS[t]:
            merged.setdefault(c, set()).add(t)

    runnable, skipped = [], []
    for c in sorted(merged, key=lambda k: (-k[0] * k[1], k[0])):
        n_mels, frames, h, b, z = c
        base = {
            "config": c,
            "cfg_name": cfg_name(c),
            "tracks": ",".join(sorted(merged[c])),
            "n_mels": n_mels,
            "frames": frames,
            "input_dim": n_mels * frames,
            "hidden": h,
            "n_blocks": b,
            "bottleneck": z,
            "tiles": model_tiles(n_mels * frames, h, b, z),
            "max_layer_tiles": max_layer_tiles(n_mels * frames, h, b, z),
        }
        target = runnable if cache_exists(base_cfg, n_mels) else skipped
        for seed in seeds:
            target.append({**base, "seed": seed, "name": run_name(c, seed)})
    return runnable, skipped


def aggregate(results_dir: Path, runnable: list[dict]) -> None:
    """收集每次运行，按配置聚合 mean ± std。"""
    rows = []
    for e in runnable:
        mj = results_dir / e["name"] / "metrics.json"
        if not mj.exists():
            continue
        m = json.loads(mj.read_text(encoding="utf-8"))
        row = {k: e[k] for k in ("cfg_name", "name", "seed", "tracks", *CFG_KEYS,
                                 "input_dim", "tiles", "max_layer_tiles")}
        row["auc"] = m["auc"]
        row["pauc"] = m["pauc"]
        hj = results_dir / e["name"] / "history.json"
        if hj.exists():
            h = json.loads(hj.read_text(encoding="utf-8"))
            row["params"] = h.get("params")
            row["wall_min"] = round(h.get("wall_seconds", 0) / 60, 2)
        rows.append(row)

    if not rows:
        print("没有可聚合的结果。")
        return

    runs = pd.DataFrame(rows)
    runs.to_csv(results_dir / "sweep_runs.csv", index=False)

    group_cols = ["cfg_name", "tracks", *CFG_KEYS, "input_dim", "tiles",
                  "max_layer_tiles", "params"]
    agg = (runs.groupby(group_cols, dropna=False)
                .agg(n_seeds=("seed", "nunique"),
                     auc_mean=("auc", "mean"), auc_std=("auc", "std"),
                     pauc_mean=("pauc", "mean"), pauc_std=("pauc", "std"),
                     wall_min=("wall_min", "mean"))
                .reset_index()
                .sort_values("input_dim", ascending=False))
    # 单种子时 std 为 NaN，填 0 但用 n_seeds 标明不可信
    agg[["auc_std", "pauc_std"]] = agg[["auc_std", "pauc_std"]].fillna(0.0)
    agg.to_csv(results_dir / "sweep.csv", index=False)

    bar = "=" * 100
    print(f"\n{bar}\n聚合结果 -> {results_dir / 'sweep.csv'}"
          f"   （每次运行明细见 sweep_runs.csv）\n{bar}")
    show = agg.copy()
    show["AUC"] = show.apply(lambda r: f"{r.auc_mean:.4f} ± {r.auc_std:.4f}", axis=1)
    show["pAUC"] = show.apply(lambda r: f"{r.pauc_mean:.4f} ± {r.pauc_std:.4f}", axis=1)
    cols = ["cfg_name", "tracks", "input_dim", "hidden", "tiles", "params",
            "n_seeds", "AUC", "pAUC"]
    print(show[cols].to_string(index=False))

    single = agg[agg.n_seeds < 2]
    if len(single):
        print(f"\n[警告] 以下 {len(single)} 个配置只有 1 个种子，误差棒未知，不要单独下结论：")
        print("        " + ", ".join(single.cfg_name))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/fast.yaml")
    ap.add_argument("--track", nargs="+", default=list(TRACKS), choices=list(TRACKS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0],
                    help="随机种子，如 --seeds 0 1 2。多种子才有误差棒")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-migrate", action="store_true", help="不迁移首轮无后缀的 run 目录")
    args = ap.parse_args()

    base = load_config(args.config)
    results_dir = Path(base.output.results_dir)

    if not args.no_migrate and results_dir.exists():
        n = migrate_legacy(results_dir)
        if n:
            print(f"迁移了 {n} 个首轮 run 目录到 _s0 命名\n")

    runnable, skipped = build_plan(base, args.track, args.seeds)

    bar = "=" * 100
    print(f"\n{bar}\n扫描计划  track={args.track}  seeds={args.seeds}  基础配置={args.config}\n{bar}")
    print(f"{'name':<24}{'tracks':<14}{'dim':>6}{'hidden':>8}{'tiles':>7}{'seed':>6}  状态")
    print("-" * 100)

    todo = []
    for e in runnable:
        done = (results_dir / e["name"] / "metrics.json").exists()
        if not done or args.force:
            todo.append(e)
        print(f"{e['name']:<24}{e['tracks']:<14}{e['input_dim']:>6}{e['hidden']:>8}"
              f"{e['tiles']:>7}{e['seed']:>6}  {'待跑' if (not done or args.force) else '已完成'}")

    for e in skipped:
        print(f"{e['name']:<24}{e['tracks']:<14}{e['input_dim']:>6}{e['hidden']:>8}"
              f"{e['tiles']:>7}{e['seed']:>6}  跳过: 缺 n_mels={e['n_mels']} 缓存")
    if skipped:
        miss = sorted({e["n_mels"] for e in skipped})
        print(f"\n补缓存：python scripts/prepare_data.py --n-mels {' '.join(map(str, miss))}")

    est = sum(2.0 if e["hidden"] <= 30 else 5.0 for e in todo)
    print(f"\n共 {len(runnable)} 次运行，其中 {len(todo)} 次待跑（约 {est:.0f} 分钟）")
    if args.dry_run:
        return 0

    for i, e in enumerate(todo, 1):
        n_mels, frames, h, b, z = e["config"]
        print(f"\n{bar}\n[{i}/{len(todo)}] {e['name']}  dim={e['input_dim']} "
              f"hidden={h} tiles={e['tiles']} seed={e['seed']}\n{bar}")
        cfg = OmegaConf.merge(base, OmegaConf.from_dotlist([
            f"name={e['name']}",
            f"seed={e['seed']}",
            f"feature.n_mels={n_mels}",
            f"feature.frames={frames}",
            f"model.hidden={h}",
            f"model.n_blocks={b}",
            f"model.bottleneck={z}",
        ]))
        t0 = time.time()
        try:
            train(cfg)
            evaluate(cfg)
        except Exception as exc:  # 单次失败不中断整个扫描
            print(f"[FAIL] {e['name']}: {type(exc).__name__}: {exc}")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print(f"[{i}/{len(todo)}] {e['name']} 用时 {(time.time() - t0) / 60:.1f} 分钟")

    aggregate(results_dir, runnable)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
