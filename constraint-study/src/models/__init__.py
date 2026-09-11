"""模型定义。Day 3 起会加入 analog.py（量化 + 噪声 + 分块）。"""

from .autoencoder import DenseAutoEncoder, build_from_config, reconstruction_error

__all__ = ["DenseAutoEncoder", "build_from_config", "reconstruction_error"]
