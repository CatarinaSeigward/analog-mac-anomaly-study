"""评估：逐文件异常分数 -> 按 machine ID 计算 AUC / pAUC。

用法::

    python -m src.evaluate --config configs/baseline.yaml
    python -m src.evaluate --config configs/fast.yaml --compare c0_baseline

★ 异常分数定义（必须与参考实现一致，否则复现不出官方数字）：
  一个文件内**所有窗口重构 MSE 的均值**。评估恒用 stride=1。

★ pAUC 用 sklearn 的 ``roc_auc_score(..., max_fpr=0.1)``，
  即 McClish 标准化的部分 AUC —— 与 DCASE / MLPerf Tiny 的定义一致。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from .features.logmel import cache_prefix, load_cache
from .models.autoencoder import DenseAutoEncoder, reconstruction_error
from .utils import ensure_dir, get_device, load_config

# MLPerf Tiny 官方参考数字（ToyCar，machine id 1-4 平均）
REFERENCE = {"auc": 0.800914, "pauc": 0.672218}


def file_scores(
    model: torch.nn.Module,
    mel: np.ndarray,
    meta: dict,
    frames: int,
    device: torch.device,
    batch: int = 16384,
    store_gpu_limit_gb: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (scores, labels, machine_ids)，每个文件一个元素。

    异常分数 = 文件内所有窗口重构 MSE 的均值（与参考实现一致）。

    实现说明（Day 3 性能优化，见 ROADMAP 2.4）：
    原先按文件做 Python 循环，2459 次小批量调用，GPU 大部分时间在等调度。
    改为把**全部文件的全部窗口**拼成一个索引数组统一批处理，再用 index_add_
    按文件做分段求和。数值上与逐文件求均值完全等价（同样是 sum/count）。
    """
    model.eval()
    n_mels = int(meta["n_mels"])

    arr = torch.as_tensor(np.ascontiguousarray(mel), dtype=torch.float32)
    store = device
    if device.type == "cuda" and arr.nbytes / 2**30 > store_gpu_limit_gb:
        store = torch.device("cpu")   # 与 WindowStore 同样的 WDDM 兜底
    mel_t = arr.to(store)
    ar = torch.arange(frames, device=store)

    # 全局窗口索引 + 每个窗口所属的文件序号（窗口不跨文件边界）
    starts_l, file_of_l, keep = [], [], []
    for fi, e in enumerate(meta["files"]):
        n_win = e["n_frames"] - frames + 1
        if n_win <= 0:
            continue
        file_of_l.append(torch.full((n_win,), len(keep), dtype=torch.long))
        keep.append(fi)
        starts_l.append(torch.arange(e["offset"], e["offset"] + n_win, dtype=torch.long))

    if not keep:
        return np.empty(0), np.empty(0), np.empty(0)

    starts = torch.cat(starts_l).to(store)
    file_of = torch.cat(file_of_l).to(device)
    n_files = len(keep)

    sums = torch.zeros(n_files, device=device)
    counts = torch.zeros(n_files, device=device)
    for i in range(0, starts.numel(), batch):
        idx = starts[i : i + batch].unsqueeze(1) + ar.unsqueeze(0)
        x = mel_t[idx].reshape(idx.shape[0], frames * n_mels)
        if store != device:
            x = x.to(device, non_blocking=True)
        err = reconstruction_error(model, x)
        f = file_of[i : i + batch]
        sums.index_add_(0, f, err)
        counts.index_add_(0, f, torch.ones_like(err))

    scores = (sums / counts).cpu().numpy()
    labels = np.array([meta["files"][fi]["label"] for fi in keep])
    mids = np.array([meta["files"][fi]["machine_id"] for fi in keep])
    return scores, labels, mids


def file_scores_reference(
    model: torch.nn.Module, mel: np.ndarray, meta: dict, frames: int,
    device: torch.device, batch: int = 4096,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """逐文件循环的朴素实现，只用于验证优化版的数值等价性。"""
    model.eval()
    n_mels = int(meta["n_mels"])
    mel_t = torch.as_tensor(np.ascontiguousarray(mel), dtype=torch.float32).to(device)
    ar = torch.arange(frames, device=device)
    scores, labels, mids = [], [], []
    for e in meta["files"]:
        n_win = e["n_frames"] - frames + 1
        if n_win <= 0:
            continue
        rows = torch.arange(e["offset"], e["offset"] + n_win, device=device)
        errs = []
        for i in range(0, n_win, batch):
            idx = rows[i : i + batch].unsqueeze(1) + ar.unsqueeze(0)
            x = mel_t[idx].reshape(idx.shape[0], frames * n_mels)
            errs.append(reconstruction_error(model, x))
        scores.append(torch.cat(errs).mean().item())
        labels.append(e["label"])
        mids.append(e["machine_id"])
    return np.asarray(scores), np.asarray(labels), np.asarray(mids)


def per_machine_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    mids: np.ndarray,
    machine_ids: list[int],
    max_fpr: float = 0.1,
) -> pd.DataFrame:
    rows = []
    for mid in machine_ids:
        m = mids == mid
        y, s = labels[m], scores[m]
        if m.sum() == 0 or len(np.unique(y)) < 2:
            rows.append({"machine_id": mid, "n": int(m.sum()), "auc": np.nan, "pauc": np.nan})
            continue
        rows.append({
            "machine_id": mid,
            "n": int(m.sum()),
            "auc": roc_auc_score(y, s),
            "pauc": roc_auc_score(y, s, max_fpr=max_fpr),
        })
    df = pd.DataFrame(rows)
    df.loc[len(df)] = {
        "machine_id": "mean",
        "n": int(df["n"].sum()),
        "auc": df["auc"].mean(skipna=True),
        "pauc": df["pauc"].mean(skipna=True),
    }
    return df


def evaluate(cfg, compare: str | None = None) -> pd.DataFrame:
    device = get_device()
    out = ensure_dir(Path(cfg.output.results_dir) / cfg.name)

    ckpt = torch.load(out / "model.pt", map_location=device, weights_only=False)
    model = DenseAutoEncoder(
        input_dim=ckpt["input_dim"],
        hidden=int(cfg.model.hidden),
        n_blocks=int(cfg.model.n_blocks),
        bottleneck=int(cfg.model.bottleneck),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])

    mel, meta = load_cache(cache_prefix(cfg.data.cache_dir, int(cfg.feature.n_mels), "test"))
    s, y, m = file_scores(model, mel, meta, int(cfg.feature.frames), device)
    df = per_machine_metrics(s, y, m, list(cfg.eval.machine_ids), float(cfg.eval.max_fpr))

    print()
    print(df.to_string(index=False, float_format=lambda v: f"{v:.6f}"))

    auc = float(df.iloc[-1]["auc"])
    pauc = float(df.iloc[-1]["pauc"])
    print()
    print(f"[参考] MLPerf Tiny 官方: AUC {REFERENCE['auc']:.6f}  pAUC {REFERENCE['pauc']:.6f}")
    print(f"[本次] AUC {auc:.6f} ({auc - REFERENCE['auc']:+.4f})  "
          f"pAUC {pauc:.6f} ({pauc - REFERENCE['pauc']:+.4f})")

    if compare:
        ref_path = Path(cfg.output.results_dir) / compare / "metrics.json"
        if ref_path.exists():
            r = json.loads(ref_path.read_text(encoding="utf-8"))
            d_auc = auc - r["auc"]
            d_pauc = pauc - r["pauc"]
            print()
            print(f"[对照] {compare}: AUC {r['auc']:.6f}  pAUC {r['pauc']:.6f}")
            print(f"[差异] dAUC {d_auc:+.4f}   dpAUC {d_pauc:+.4f}")
            verdict = ("OK  |dAUC| < 0.01，与对照组等价，可用于扫描"
                       if abs(d_auc) < 0.01 else
                       "WARN |dAUC| >= 0.01，不能替代对照组")
            print(f"[判定] {verdict}")
        else:
            print()
            print(f"[对照] 找不到 {ref_path}")
    elif 0.78 <= auc <= 0.82:
        print("[验收] OK 落在 0.78-0.82，Day 1 通过")
    else:
        print("[验收] FAIL 偏离过大 —— 先排查，不要继续。常见原因见 根目录 README.md 的 Data 一节")

    df.to_csv(out / "metrics.csv", index=False)
    payload = {"auc": auc, "pauc": pauc, "reference": REFERENCE,
               "n_mels": int(cfg.feature.n_mels), "frames": int(cfg.feature.frames),
               "input_dim": int(ckpt["input_dim"])}
    with open(out / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("-o", "--override", nargs="*", default=[])
    ap.add_argument("--compare", default=None,
                    help="与 results/<name>/metrics.json 对照，如 c0_baseline")
    args = ap.parse_args()
    evaluate(load_config(args.config, args.override), compare=args.compare)


if __name__ == "__main__":
    main()
