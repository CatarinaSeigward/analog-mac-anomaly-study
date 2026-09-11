"""Day 0 验收脚本：检查环境是否就绪。

    python scripts/check_env.py

全绿才往下走。aihwkit 那一项现在红的没关系（Day 3 才需要，且有 fallback）。
"""

from __future__ import annotations

import importlib
import platform
import sys

OK, BAD, WARN = "[ OK ]", "[FAIL]", "[WARN]"

REQUIRED = [
    ("numpy", None),
    ("scipy", None),
    ("torch", None),
    ("librosa", None),
    ("soundfile", None),
    ("sklearn", "scikit-learn"),
    ("omegaconf", None),
    ("yaml", "pyyaml"),
    ("matplotlib", None),
    ("pandas", None),
    ("tqdm", None),
]

OPTIONAL = [
    ("aihwkit", "aihwkit（Day 3 才需要；Windows 上装不上就走 fallback，见 PLAN 附录 I）"),
    ("torchaudio", "torchaudio（librosa 的备选特征后端）"),
]


def _ver(mod) -> str:
    return getattr(mod, "__version__", "?")


def main() -> int:
    print(f"Python  {platform.python_version()}  ({sys.executable})")
    print(f"平台    {platform.platform()}\n")

    failed = []
    print("--- 必需依赖 ---")
    for name, pip_name in REQUIRED:
        try:
            m = importlib.import_module(name)
            print(f"{OK} {name:<12} {_ver(m)}")
        except Exception as e:
            failed.append(pip_name or name)
            print(f"{BAD} {name:<12} {type(e).__name__}: {e}")

    print("\n--- 可选依赖 ---")
    for name, note in OPTIONAL:
        try:
            m = importlib.import_module(name)
            print(f"{OK} {name:<12} {_ver(m)}")
        except Exception:
            print(f"{WARN} {name:<12} 未安装 — {note}")

    print("\n--- CUDA ---")
    try:
        import torch

        if torch.cuda.is_available():
            print(f"{OK} CUDA 可用: {torch.cuda.get_device_name(0)}")
            free, total = torch.cuda.mem_get_info()
            print(f"       显存 {total/2**30:.1f} GB（空闲 {free/2**30:.1f} GB）")
            if total / 2**30 < 4:
                print(f"{WARN} 显存 < 4 GB，训练时把 log-mel 缓存留在 CPU（见 train.py WindowStore）")
        else:
            print(f"{WARN} CUDA 不可用 —— 装的可能是 CPU-only 的 torch。")
            print("       重装：pip install torch --index-url https://download.pytorch.org/whl/cu126")
            print("       （CPU 也能跑，只是慢；不阻塞 Day 0 验收）")
    except Exception as e:
        print(f"{BAD} 无法检查 CUDA: {e}")

    print()
    if failed:
        print(f"{BAD} 缺少必需依赖: {', '.join(sorted(set(failed)))}")
        print("     pip install -r requirements.txt")
        return 1
    print(f"{OK} Day 0 环境验收通过。下一步：按 根目录 README.md 的 Data 一节下载数据，然后跑 scripts/prepare_data.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
