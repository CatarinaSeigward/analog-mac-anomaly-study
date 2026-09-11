#!/usr/bin/env bash
# Day 4 补跑：按修正后的训练配方（训练时 σ_read = 4%）重做比特扫描。
#
# 第一次比特扫描用 σ_read = 0 训练，被 none 架构的配方问题污染（见 sweep_bits.py 说明）。
# 原 checkpoint 与 results/bits.csv 保留；本次产出 results/bits_r0.04.csv。
#
#   PY=/path/to/python bash scripts/run_day4b.sh
set -u
cd "$(dirname "$0")/.."
PY="${PY:-python}"
N="${N_SHARDS:-4}"
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
mkdir -p logs

echo "[$(date +%H:%M)] 启动 $N 个比特扫描分片（训练 σ_read = 0.04）"
for k in $(seq 0 $((N - 1))); do
  "$PY" scripts/sweep_bits.py --train-sigma-read 0.04 --shard "$k/$N" \
    > "logs/day4b_bits_shard_$k.log" 2>&1 &
done
wait

echo "[$(date +%H:%M)] 训练结束，开始统一评估"
"$PY" scripts/sweep_bits.py --train-sigma-read 0.04 --eval-only > logs/day4b_eval_bits.log 2>&1
echo "[$(date +%H:%M)] 完成"
