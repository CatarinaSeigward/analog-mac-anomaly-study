"""通用工具：配置加载、随机种子、设备选择。"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf


def load_config(path: str | Path, overrides: list[str] | None = None) -> DictConfig:
    """读 YAML 配置，可选地用命令行式的 dotlist 覆盖（如 ``feature.n_mels=32``）。"""
    cfg = OmegaConf.load(Path(path))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return cfg  # type: ignore[return-value]


def set_seed(seed: int) -> None:
    """固定所有随机源。可复现性是本研究的信用基础，见 PLAN §9。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
