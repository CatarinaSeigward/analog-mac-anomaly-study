"""检查数据目录布局，统计样本数，构建 log-mel 缓存。

    python scripts/prepare_data.py                      # 用配置里的 n_mels
    python scripts/prepare_data.py --n-mels 128 64 32 30 16   # 批量建多份缓存
    python scripts/prepare_data.py --limit 50           # 冒烟测试
    python scripts/prepare_data.py --check-only         # 只检查布局，不提特征

缓存按 n_mels 分目录：cache/n_mels{N}/{train,test}_logmel.npy
mel 滤波器组不能从 128 带降采样得到，所以每个 n_mels 值必须单独提取一次。
frames 不影响缓存（只是窗口宽度），改 frames 无需重建。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dcase import list_files, summarize  # noqa: E402
from src.features.logmel import build_cache, cache_prefix  # noqa: E402
from src.utils import load_config  # noqa: E402

EXPECTED = {"train": 7000, "test": 2459}


def report(split: str, files) -> None:
    stats = summarize(files)
    print(f"\n[{split}] 共 {len(files)} 个文件")
    print(f"  {'machine_id':<12}{'normal':>8}{'anomaly':>9}")
    for mid, c in stats.items():
        print(f"  {mid:<12}{c['normal']:>8}{c['anomaly']:>9}")
    exp = EXPECTED.get(split)
    if exp:
        delta = len(files) - exp
        ok = abs(delta) <= exp * 0.02
        print(f"  {'OK ' if ok else 'WARN'} 期望约 {exp}，实际 {len(files)}（{delta:+d}）")
        if not ok:
            print("       数量对不上通常是漏下了 dc2020task2added，见 根目录 README.md 的 Data 一节")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--n-mels", type=int, nargs="+", default=None,
                    help="要构建缓存的 n_mels 值，可多个。默认用配置里的值")
    ap.add_argument("--limit", type=int, default=None, help="每个划分只处理前 N 个文件")
    ap.add_argument("--force", action="store_true", help="缓存已存在也重建")
    ap.add_argument("--check-only", action="store_true", help="只检查布局，不提特征")
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = Path(cfg.data.root)
    n_mels_list = args.n_mels or [int(cfg.feature.n_mels)]

    splits = {"train": list(cfg.data.train_dirs), "test": list(cfg.data.test_dirs)}
    found: dict[str, list] = {}

    for split, subdirs in splits.items():
        try:
            files = list_files(root, subdirs)
        except FileNotFoundError as e:
            print(f"\n[FAIL] [{split}] {e}")
            return 1
        report(split, files)
        found[split] = files[: args.limit] if args.limit else files

    if args.check_only:
        print("\n[OK] 目录布局检查通过（--check-only，未提取特征）")
        return 0

    print(f"\n将为 n_mels = {n_mels_list} 构建缓存")
    for n_mels in n_mels_list:
        print(f"\n{'=' * 60}\nn_mels = {n_mels}\n{'=' * 60}")
        for split, files in found.items():
            prefix = cache_prefix(cfg.data.cache_dir, n_mels, split)
            if Path(f"{prefix}_logmel.npy").exists() and not args.force:
                print(f"  [{split}] 缓存已存在，跳过（要重建加 --force）")
                continue
            data, meta = build_cache(
                files, prefix,
                n_mels=n_mels,
                n_fft=int(cfg.feature.n_fft),
                hop_length=int(cfg.feature.hop_length),
                power=float(cfg.feature.power),
                backend=str(cfg.feature.backend),
                desc=f"n_mels{n_mels} {split}",
            )
            n_win = sum(max(0, e["n_frames"] - int(cfg.feature.frames) + 1)
                        for e in meta["files"])
            print(f"  [{split}] {data.shape} float32, {data.nbytes / 2**30:.2f} GB, "
                  f"{n_win:,} 个窗口 -> {prefix}")

    print("\n[OK] 数据准备完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
