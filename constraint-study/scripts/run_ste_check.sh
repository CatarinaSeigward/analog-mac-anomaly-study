#!/usr/bin/env bash
# STE 修复后的小规模验证（2026-09-22）：4/5-bit 的"崩溃"是硬件下限，还是量化训练的 bug？
#
# 旧 quantize_ste 让取整到最高电平的元素没有梯度（见 src/models/quant.py 的 _RoundClampSTE），
# 输出层 bias 在全部 QAT 训练中基本冻结。本脚本用修复后的代码重训：
#
#   W/S/B 同步 4 / 5 / 6 bit   9 个任务  -> results/bits_r0.04_b4-5-6_stefix.csv
#   只扫 W，4 / 5 bit          6 个任务  -> results/bits_r0.04_w_b4-5_stefix.csv
#
# 对照是旧代码训练的同名 run（不带 _stefix），checkpoint 原样保留。
# 前向没有变：旧 checkpoint 的评估结果不受修复影响，变的只有训练。
#
# 内存：每个训练进程约 1.4 GB 主机内存，按空闲物理内存定 N_SHARDS（见 run_bits_split.sh）。
#
#   PY=/path/to/python N_SHARDS=2 bash scripts/run_ste_check.sh
set -u
cd "$(dirname "$0")/.."
PY="${PY:-python}"
N="${N_SHARDS:-2}"
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
mkdir -p logs

CMDS=("--bits 4 5 6 --tag stefix" "--sweep-only w --bits 4 5 --tag stefix")

echo "[$(date +%H:%M)] 启动 $N 个训练分片"
for k in $(seq 0 $((N - 1))); do
  (
    for c in "${!CMDS[@]}"; do
      # shellcheck disable=SC2086
      "$PY" scripts/sweep_bits.py ${CMDS[$c]} --shard "$(( (k + c) % N ))/$N"
    done
  ) >| "logs/ste_check_shard_$k.log" 2>&1 &
done
wait

echo "[$(date +%H:%M)] 训练结束，开始统一评估"
for c in "${!CMDS[@]}"; do
  # shellcheck disable=SC2086
  "$PY" scripts/sweep_bits.py ${CMDS[$c]} --eval-only
done >| logs/ste_check_eval.log 2>&1
echo "[$(date +%H:%M)] 完成"
