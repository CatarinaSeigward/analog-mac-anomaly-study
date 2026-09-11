"""模拟器件非理想性的参数与随机源管理。

★ 本模块最重要的职责是把 **D2D 与 C2C 分开**，以及把**芯片采样的随机源
   与训练随机源隔离**。这两件事搞错会让整个 Day 3 的结论失去意义。

D2D（device-to-device，器件间失配）
    每颗芯片的固有偏差。**采样一次然后冻结**，可以通过逐片标定（BN 的 γ/β）补偿。
    评估时：采样 N 颗芯片，每颗冻结自己的 D2D，报跨芯片的 mean ± std。
    训练时：每个 batch 重采样，让模型对**任意**芯片鲁棒
            （部署芯片的 D2D 在训练时是未知的）。

C2C（cycle-to-cycle，周期间噪声）
    热噪声、闪烁噪声等。**每次推理都重新随机**，不可标定。

随机源隔离
    - 芯片采样用**独立的 generator**（`chip_generator()`），种子与训练种子无关。
      否则"芯片编号"会和"训练种子"混杂，方差归因就错了。
    - 训练噪声与数据打乱共用训练种子 —— 这是**有意的**：两者都属于
      "同一次训练运行的随机性"，本来就该一起变。

参数来源
    这里的默认值取自 PCM/ReRAM 文献的常见区间，**不是 Ai Linear 的实测数字**。
    真实参数应由模拟设计团队提供（见 assumptions.md 的 A2/A6/A15）。
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

ADC_MODES = ("every_input", "per_layer", "none")
"""层边界上 A/D、D/A 的放置方式，由 ``models.analog.apply_adc_mode`` 按层位置应用。

``"every_input"``  每层**输入**量化。Day 3 的实现，保留仅为原样复现 Day 3。
``"per_layer"``    每层输入 D/A + 输出 A/D                             —— 常规 CIM
``"none"``         仅第一层输入 D/A、最后一层输出 A/D，隐藏层全程模拟  —— Ai Linear
"""


@dataclass(frozen=True)
class DeviceParams:
    """器件非理想性参数。

    位宽字段 ``<= 0`` 表示**关闭该项量化**（用于退化测试与 fp32 对照组）。
    """

    # --- 量化（芯片规格：W/S/B 各 6-bit）---
    w_bits: int = 6            # 权重
    s_bits: int = 6            # 激活/信号（层边界上的 D/A、A/D）
    b_bits: int = 6            # 偏置

    # --- 噪声 ---
    sigma_prog: float = 0.03   # 权重编程误差（乘性），来自 program-verify 容差带
    sigma_d2d: float = 0.02    # 器件间失配（逐芯片固定，可标定）
    sigma_read: float = 0.0    # 信号通路读噪声（加性），std = sigma_read × mean|W·x|

    # --- 其他 ---
    clip: float = 0.0          # 激活饱和点，0 = 不裁剪
    drift: float = 0.0         # 保持性漂移（EEPROM/ReRAM 机制不同，见 assumptions A2）
    tile: int = 30             # 阵列尺寸，决定 per-tile 量化尺度的粒度

    # --- 架构开关 ---
    adc_mode: str = "every_input"
    """层边界上的 A/D、D/A 放置，取值见 ``ADC_MODES``。"""

    requantize_output: bool = False
    """每层输出是否再量化一次。**仅在 ``adc_mode="every_input"`` 下生效。**

    ⚠️ 更正（Day 4 审稿）：Day 3 曾把 ``False`` 标注为 "Ai Linear 建模"，这是错的。
    every_input 模式下每层**输入**都会量化，每个层边界都有一次 D/A ——
    真正的"隐藏层无 A/D、D/A"要用 ``adc_mode="none"``。
    """

    def __post_init__(self) -> None:
        if self.adc_mode not in ADC_MODES:
            raise ValueError(f"未知 adc_mode: {self.adc_mode!r}，可选 {ADC_MODES}")

    @property
    def noise_free(self) -> bool:
        return self.sigma_prog == 0 and self.sigma_d2d == 0 and self.sigma_read == 0

    def with_(self, **kw) -> "DeviceParams":
        """返回改了若干字段的新实例（frozen dataclass 的便捷改法）。"""
        return replace(self, **kw)


# 常用预设 ---------------------------------------------------------------

IDEAL = DeviceParams(w_bits=0, s_bits=0, b_bits=0,
                     sigma_prog=0.0, sigma_d2d=0.0, sigma_read=0.0)
"""完全理想：无量化无噪声。AnalogLinear 在此配置下必须等价于 nn.Linear。"""

QUANT_ONLY = DeviceParams(sigma_prog=0.0, sigma_d2d=0.0, sigma_read=0.0)
"""只有 6-bit 量化，无噪声。用于隔离量化与噪声各自的贡献。"""


# 随机源 -----------------------------------------------------------------

CHIP_SEED_BASE = 20260910
"""芯片采样的基准种子。★ 与训练种子无关，保证芯片编号不与训练随机性混杂。"""


def chip_generator(chip_id: int, base: int = CHIP_SEED_BASE) -> torch.Generator:
    """返回第 ``chip_id`` 颗芯片专用的 CPU generator。

    用 CPU generator 是为了跨设备可复现（CUDA 的 RNG 在不同架构上不保证一致）。
    D2D 只在 ``set_chip()`` 时采样一次，CPU 采样的开销可以忽略。
    """
    g = torch.Generator()
    g.manual_seed(base + int(chip_id))
    return g


def sample_d2d(shape: tuple[int, ...], sigma_d2d: float,
               generator: torch.Generator | None = None,
               device: torch.device | str = "cpu") -> torch.Tensor:
    """采样一颗芯片的 D2D 乘性因子，形状同权重。

    返回 ``1 + N(0, sigma_d2d)``；``sigma_d2d == 0`` 时返回全 1。
    """
    if sigma_d2d <= 0:
        return torch.ones(shape, device=device)
    noise = torch.randn(shape, generator=generator)   # 在 CPU 上采样保证可复现
    return (1.0 + noise * sigma_d2d).to(device)
