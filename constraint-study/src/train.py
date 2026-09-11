"""训练 C0 基线自编码器。

用法::

    python -m src.train --config configs/baseline.yaml
    python -m src.train --config configs/baseline.yaml -o train.epochs=5   # 冒烟测试

设计说明：log-mel 缓存整体常驻显存（约 1.1 GB），窗口用 gather 现取。
这样绕开 DataLoader，避免小样本训练里数据加载成为瓶颈。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from .features.logmel import cache_prefix, load_cache, window_start_rows
from .models.analog import analogize, count_tiles
from .models.autoencoder import build_from_config
from .noise import DeviceParams
from .utils import count_params, ensure_dir, get_device, load_config, set_seed


class WindowStore:
    """log-mel 常驻某设备，按窗口起始行 gather 出 [B, frames*n_mels]。

    ``store_device`` 决定缓存放哪：
      - ``"cuda"``  常驻显存，零传输，最快
      - ``"cpu"``   常驻内存，每步传一个 batch（2048x640 float32 约 5 MB，开销可忽略）
      - ``"auto"``  小于 ``gpu_limit_gb`` 才放显存，否则放内存

    为什么需要 auto：8 GB 的笔记本 GPU 在 WDDM 下，即使 nvidia-smi 显示 7 GB 空闲，
    单次 1.15 GB 的大块分配也可能被拒（进程显存预算受其他 GPU 程序挤占）。
    与其撞 OOM 再兜底（失败的分配会让 CUDA 上下文不稳），不如按大小提前决定。
    """

    def __init__(
        self,
        mel: np.ndarray,
        starts: np.ndarray,
        frames: int,
        n_mels: int,
        device: torch.device,
        store_device: str = "auto",
        gpu_limit_gb: float = 0.5,
    ) -> None:
        self.frames = frames
        self.n_mels = n_mels
        self.device = device

        arr = torch.as_tensor(np.ascontiguousarray(mel), dtype=torch.float32)
        size_gb = arr.nbytes / 2**30

        if device.type != "cuda" or store_device == "cpu":
            self.store = torch.device("cpu")
        elif store_device == "cuda":
            self.store = device
        else:  # auto
            self.store = device if size_gb <= gpu_limit_gb else torch.device("cpu")

        if self.store.type == "cpu" and device.type == "cuda":
            print(f"[store] log-mel {size_gb:.2f} GB 常驻 CPU，逐 batch 传输 "
                  f"(store_device={store_device}, 阈值 {gpu_limit_gb} GB)")
        else:
            print(f"[store] log-mel {size_gb:.2f} GB 常驻 {self.store}")

        self.mel = arr.to(self.store)
        self.starts = torch.as_tensor(starts, dtype=torch.long).to(self.store)
        self._ar = torch.arange(frames, device=self.store)

    def __len__(self) -> int:
        return int(self.starts.numel())

    @property
    def dim(self) -> int:
        return self.frames * self.n_mels

    def gather(self, idx: torch.Tensor) -> torch.Tensor:
        """idx 是窗口下标（在 store 设备上）。返回计算设备上的 [B, frames*n_mels]。"""
        rows = self.starts[idx].unsqueeze(1) + self._ar.unsqueeze(0)
        x = self.mel[rows].reshape(idx.numel(), self.dim)
        return x.to(self.device, non_blocking=True) if self.store != self.device else x


def train(cfg, params: "DeviceParams | None" = None) -> dict:
    """训练自编码器。

    ``params`` 为 None 时是普通 fp32 训练（C1/C2 用）。
    传入 DeviceParams 时把模型的 nn.Linear 换成 AnalogLinear，
    训练全程注入量化与噪声 —— 这就是硬件感知训练（HWA，C3/C4 用）。
    """
    set_seed(int(cfg.seed))
    device = get_device()
    print(f"[device] {device}")

    # ---------------- 数据 ----------------
    prefix = cache_prefix(cfg.data.cache_dir, int(cfg.feature.n_mels), "train")
    mel, meta = load_cache(prefix)
    frames, n_mels = int(cfg.feature.frames), int(meta["n_mels"])
    stride = int(cfg.train.get("window_stride", 1))
    starts = window_start_rows(meta, frames, stride)

    store = WindowStore(
        mel, starts, frames, n_mels, device,
        store_device=str(cfg.train.get("store_device", "auto")),
        gpu_limit_gb=float(cfg.train.get("gpu_limit_gb", 0.5)),
    )
    n_total = len(store)
    print(f"[data] {len(meta['files'])} 个文件, {n_total:,} 个窗口"
          f"{f' (stride={stride})' if stride > 1 else ''}, 输入维度 {store.dim}")

    # 训练 / 验证切分
    g = torch.Generator().manual_seed(int(cfg.seed))
    perm = torch.randperm(n_total, generator=g)
    n_val = int(n_total * float(cfg.train.validation_split))
    val_idx = perm[:n_val].to(store.store)
    train_idx = perm[n_val:].to(store.store)
    print(f"[data] train {len(train_idx):,} / val {len(val_idx):,}")

    # ---------------- 模型 ----------------
    model = build_from_config(cfg, input_dim=store.dim)
    if params is not None:
        analogize(model, params)
        print(f"[model] HWA 训练: W/S/B={params.w_bits}/{params.s_bits}/{params.b_bits}bit, "
              f"sigma_prog={params.sigma_prog}, sigma_d2d={params.sigma_d2d}, "
              f"sigma_read={params.sigma_read}, adc_mode={params.adc_mode}, "
              f"tiles={count_tiles(model)}")
    model = model.to(device)
    print(f"[model] 参数量 {count_params(model):,}")
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg.train.lr))

    bs = int(cfg.train.batch_size)
    epochs = int(cfg.train.epochs)
    history: list[dict] = []
    t0 = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        order = train_idx[torch.randperm(len(train_idx), device=store.store)]
        tot, nb = 0.0, 0
        for i in range(0, len(order), bs):
            x = store.gather(order[i : i + bs])
            loss = F.mse_loss(model(x), x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        tr = tot / max(nb, 1)

        model.eval()
        with torch.no_grad():
            vt, vb = 0.0, 0
            for i in range(0, len(val_idx), bs):
                x = store.gather(val_idx[i : i + bs])
                vt += F.mse_loss(model(x), x).item()
                vb += 1
            va = vt / max(vb, 1)

        history.append({"epoch": ep, "train_loss": tr, "val_loss": va})
        if ep == 1 or ep % 10 == 0 or ep == epochs:
            print(f"[{ep:>3}/{epochs}] train {tr:.6f}  val {va:.6f}  ({time.time()-t0:.0f}s)")

    # ---------------- 保存 ----------------
    out = ensure_dir(Path(cfg.output.results_dir) / cfg.name)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": store.dim,
            "config": OmegaConf.to_container(cfg, resolve=True),
            "analog_params": params.__dict__ if params is not None else None,
        },
        out / "model.pt",
    )
    wall = time.time() - t0
    with open(out / "history.json", "w", encoding="utf-8") as fh:
        json.dump({"wall_seconds": wall, "n_windows": n_total,
                   "window_stride": stride, "epochs": epochs,
                   "batch_size": bs, "lr": float(cfg.train.lr),
                   "params": count_params(model), "history": history}, fh, indent=2)
    print(f"[saved] {out/'model.pt'}  (训练耗时 {wall/60:.1f} 分钟)")

    return {"history": history, "out_dir": str(out), "wall_seconds": wall}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("-o", "--override", nargs="*", default=[], help="如 train.epochs=5")
    args = ap.parse_args()
    train(load_config(args.config, args.override))


if __name__ == "__main__":
    main()
