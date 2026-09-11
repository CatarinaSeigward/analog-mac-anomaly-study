"""模拟 MAC 阵列的行为级仿真层。

单层前向路径（对应 Ai Linear 公开的 30x30 / 6-bit NN IC）::

    x  --[D/A：量化 S，若本层 quantize_input]-->  x_q
    W  --量化(W, per-tile)--> 乘 D2D（逐芯片固定）--> 乘 (1+N(0,σ_prog))  -->  W_eff
    b  --量化(B)-->  b_q

    y = x_q @ W_eff^T + b_q
      --加读噪声 N(0, (σ_read · mean|W_eff·x_q|)²)-->
      --饱和-->
      --[A/D：量化，若本层 quantize_output]-->  y

层边界上的 A/D、D/A 放置由 ``DeviceParams.adc_mode`` 决定，按层在网络中的位置
应用（:func:`apply_adc_mode`）::

    every_input  每层输入量化（Day 3 的实现，仅为原样复现 Day 3 保留）
    per_layer    每层输入 D/A + 输出 A/D                             —— 常规 CIM
    none         仅第一层输入 D/A、最后一层输出 A/D，隐藏层全程模拟  —— Ai Linear

⚠️ 更正（Day 4 审稿）：Day 3 的实现在**每一层的输入**都量化，所以 Day 3 标为
   "C3 / Ai Linear 建模"的配置每个层边界都有一次 D/A；真正的"隐藏层无 A/D、D/A"
   从 ``adc_mode="none"`` 才开始被仿真。见 tests/test_analog.py 里的回归测试。

★ 退化性质：``DeviceParams`` 为 IDEAL（无量化无噪声）时，本层必须与
   ``nn.Linear`` 数值一致。这是所有模拟层测试的地基。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..noise import ADC_MODES, DeviceParams, chip_generator, sample_d2d
from ..tiling import layer_tiles
from .quant import LSQQuantizer, quantize_ste, quantize_weight_per_tile


class AnalogLinear(nn.Module):
    """带量化、编程噪声、D2D/C2C、读噪声与 per-tile 尺度的全连接层。

    芯片状态由 :meth:`set_chip` 控制：

      - ``chip_id=None``（默认）—— **训练模式**：D2D 每次前向重采样，
        让模型对任意芯片鲁棒（部署芯片的 D2D 在训练时未知）。
      - ``chip_id=k`` —— **部署模式**：冻结为第 k 颗仿真芯片的 D2D。

    层边界上的 A/D、D/A 由 ``quantize_input`` / ``quantize_output`` 控制，
    通常不直接设置，而是由 :func:`apply_adc_mode` 按层位置统一设置。
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        params: DeviceParams | None = None,
        bias: bool = True,
        use_lsq: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.params = params or DeviceParams()
        self.use_lsq = use_lsq

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        # D2D：逐芯片固定 -> 必须是 buffer 而非 Parameter（不能被优化器更新）。
        # persistent=False：D2D 由 chip_id 派生，不属于模型权重，不该进 state_dict。
        # 否则 fp32 checkpoint 无法加载进模拟模型（缺少 d2d 键），
        # 而且会把"仿真芯片编号"错误地固化进 checkpoint。
        self.register_buffer("d2d", torch.ones(out_features, in_features),
                             persistent=False)
        self._chip_id: int | None = None

        # 层边界上的 A/D、D/A。单独构造时的默认值复现 Day 3：
        # 输入量化；输出是否量化看 params.requantize_output。
        self.quantize_input: bool = True
        self.quantize_output: bool | None = None

        # LSQ 只用于激活；权重走 per-tile 静态尺度（更贴近电导映射的物理约束）
        self.act_q = LSQQuantizer(self.params.s_bits) if use_lsq else None

    # ------------------------------------------------------------------
    # 芯片状态
    # ------------------------------------------------------------------
    def set_chip(self, chip_id: int | None) -> None:
        """冻结为第 ``chip_id`` 颗仿真芯片；``None`` 回到训练模式。"""
        self._chip_id = chip_id
        if chip_id is None:
            self.d2d.fill_(1.0)
            return
        g = chip_generator(chip_id)
        self.d2d.copy_(
            sample_d2d(tuple(self.weight.shape), self.params.sigma_d2d,
                       generator=g, device=self.weight.device)
        )

    @property
    def chip_id(self) -> int | None:
        return self._chip_id

    @property
    def tiles(self) -> int:
        return layer_tiles(self.out_features, self.in_features, self.params.tile)

    # ------------------------------------------------------------------
    # 前向
    # ------------------------------------------------------------------
    def effective_weight(self) -> torch.Tensor:
        """量化 + D2D + 编程噪声之后的等效权重。"""
        p = self.params
        w = quantize_weight_per_tile(self.weight, p.w_bits, p.tile)

        if p.sigma_d2d > 0:
            if self._chip_id is None:
                # 训练模式：每次前向重采样，让模型对任意芯片鲁棒
                w = w * (1.0 + torch.randn_like(w) * p.sigma_d2d)
            else:
                w = w * self.d2d

        if p.sigma_prog > 0:
            # 编程误差 / C2C：每次前向重采样，不可标定
            w = w * (1.0 + torch.randn_like(w) * p.sigma_prog)

        if p.drift > 0:
            w = w * (1.0 - p.drift)
        return w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.params

        # 1) 输入 D/A（S）—— 只有 quantize_input 的层才量化
        if self.quantize_input:
            x = self.act_q(x) if self.act_q is not None else quantize_ste(x, p.s_bits)

        # 2) 权重（含 D2D / 编程噪声）
        w = self.effective_weight()

        # 3) 偏置量化（B）
        b = quantize_ste(self.bias, p.b_bits) if self.bias is not None else None

        y = F.linear(x, w, b)

        # 4) 读噪声：加性，按本层信号量级 mean|W·x|（不含偏置）归一
        if p.sigma_read > 0:
            signal = y - b if b is not None else y
            ref = signal.detach().abs().mean().clamp(min=1e-12)
            y = y + torch.randn_like(y) * (p.sigma_read * ref)

        # 5) 动态范围饱和
        if p.clip > 0:
            y = torch.clamp(y, -p.clip, p.clip)

        # 6) 输出 A/D —— 由 apply_adc_mode 设置；未设置时沿用 Day 3 的开关
        q_out = p.requantize_output if self.quantize_output is None else self.quantize_output
        if q_out:
            y = quantize_ste(y, p.s_bits)
        return y

    def extra_repr(self) -> str:
        p = self.params
        return (f"in={self.in_features}, out={self.out_features}, tiles={self.tiles}, "
                f"W/S/B={p.w_bits}/{p.s_bits}/{p.b_bits}bit, "
                f"sig_prog={p.sigma_prog}, sig_d2d={p.sigma_d2d}, sig_read={p.sigma_read}, "
                f"q_in={self.quantize_input}, q_out={self.quantize_output}, chip={self._chip_id}")


# ---------------------------------------------------------------------------
# 模型级工具
# ---------------------------------------------------------------------------
def analog_layers(model: nn.Module) -> list[AnalogLinear]:
    """按注册顺序返回模型中的全部 AnalogLinear。"""
    return [m for m in model.modules() if isinstance(m, AnalogLinear)]


def apply_adc_mode(model: nn.Module, mode: str) -> None:
    """按层在前向中的位置放置 A/D、D/A。

    假设 ``model.modules()`` 的顺序就是前向顺序 —— 对 DenseAutoEncoder 成立
    （encoder → decoder → out 依次注册）。换网络结构时必须重新核对。
    """
    if mode not in ADC_MODES:
        raise ValueError(f"未知 adc_mode: {mode!r}，可选 {ADC_MODES}")
    layers = analog_layers(model)
    last = len(layers) - 1
    for i, layer in enumerate(layers):
        if mode == "every_input":
            layer.quantize_input, layer.quantize_output = True, None
        elif mode == "per_layer":
            layer.quantize_input, layer.quantize_output = True, True
        else:  # none
            layer.quantize_input, layer.quantize_output = (i == 0), (i == last)


def set_chip(model: nn.Module, chip_id: int | None) -> None:
    """把整个模型里所有 AnalogLinear 切到同一颗仿真芯片。"""
    for m in analog_layers(model):
        m.set_chip(chip_id)


def set_params(model: nn.Module, params: DeviceParams) -> None:
    """批量替换器件参数，并按 ``params.adc_mode`` 重新放置 A/D、D/A。"""
    for m in analog_layers(model):
        m.params = params
    apply_adc_mode(model, params.adc_mode)


def count_tiles(model: nn.Module) -> int:
    return sum(m.tiles for m in analog_layers(model))


def _replace_linear(model: nn.Module, params: DeviceParams) -> None:
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Linear):
            al = AnalogLinear(child.in_features, child.out_features, params,
                              bias=child.bias is not None)
            al.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                al.bias.data.copy_(child.bias.data)
            al.to(child.weight.device)
            setattr(model, name, al)
        else:
            _replace_linear(child, params)


def analogize(model: nn.Module, params: DeviceParams) -> nn.Module:
    """把模型里的 ``nn.Linear`` 原地替换为 ``AnalogLinear``（权重继承），
    并按 ``params.adc_mode`` 放置 A/D、D/A。"""
    _replace_linear(model, params)
    apply_adc_mode(model, params.adc_mode)
    return model
