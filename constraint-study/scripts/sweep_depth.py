"""[已被 sweep_arch.py 取代] Day 4 深度扫描草稿。

审稿时发现本草稿的对照不成立（详见 scripts/sweep_arch.py 顶部的"审稿结论"）：

  - Day 3 的实现在每一层输入都量化，"C3 = 无层间 A/D"从未被仿真；
  - σ_read = 0 时层间 A/D 没有可再生的噪声，两种架构的比较注定无差异；
  - "同一 checkpoint 两种推理方式"在放置修正后与 HWA 的前提（训练-部署匹配）冲突。

深度扫描现在是 ``sweep_arch.py --part B``。
"""

raise SystemExit("本脚本已被 scripts/sweep_arch.py 取代，见文件说明。")
