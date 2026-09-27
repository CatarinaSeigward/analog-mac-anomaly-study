#!/usr/bin/env bash
# D/A 前去直流能否降低转换器（S）的位宽要求（2026-09-22，NEXT-STEPS §3.1 的前置检验）。
#
# STE 修复后，W 降到 4 bit 只损失约 0.01，但 S（首尾两个转换器）降到 4 bit 时 AUC 约 0.55。
# 机制猜测：log-mel 的直流约 -26 dB，起伏约 3.5 dB，max-abs 量程大半花在直流上。
# 每 batch 的 max|x| 去直流后从约 59 降到约 20（约 1.5 bit），4-bit LSB 从 8.4 dB 降到 2.9 dB。
#
#   只降 S，4 / 5 bit，去直流      6 个  -> results/bits_r0.04_s_b4-5_stefix_c.csv
#   W/S/B 同步 4 / 6 bit，去直流   6 个  -> results/bits_r0.04_b4-6_stefix_c.csv
#
# 对照是同代码、未去直流的 *_stefix run。
#
#   PY=/path/to/python N_SHARDS=3 bash scripts/run_center_check.sh
set -u
cd "$(dirname "$0")/.."
PY="${PY:-python}"
N="${N_SHARDS:-2}"
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
mkdir -p logs

CMDS=("--sweep-only s --bits 4 5 --tag stefix_c --center" "--bits 4 6 --tag stefix_c --center")

echo "[$(date +%H:%M)] 启动 $N 个训练分片"
for k in $(seq 0 $((N - 1))); do
  (
    for c in "${!CMDS[@]}"; do
      # shellcheck disable=SC2086
      "$PY" scripts/sweep_bits.py ${CMDS[$c]} --shard "$(( (k + c) % N ))/$N"
    done
  ) >| "logs/center_check_shard_$k.log" 2>&1 &
done
wait

echo "[$(date +%H:%M)] 训练结束，开始统一评估"
for c in "${!CMDS[@]}"; do
  # shellcheck disable=SC2086
  "$PY" scripts/sweep_bits.py ${CMDS[$c]} --eval-only
done >| logs/center_check_eval.log 2>&1
echo "[$(date +%H:%M)] 完成"
