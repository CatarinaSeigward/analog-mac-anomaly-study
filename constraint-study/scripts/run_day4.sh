#!/usr/bin/env bash
# Day 4 全量运行：N 个进程并行训练，全部结束后统一评估。
#
# 每个进程按优先级依次跑自己那一份：
#   Part A（图 3A，主结果） -> 比特扫描（图 4） -> Part B（图 3B，深度）
# 这样即使中途停掉，最重要的结果也已经在前面跑完。
#
# 为什么并行：模型只有几千个参数，训练被 kernel 启动开销主导，单进程 GPU 利用率
# 约 40%。实测 2 / 3 / 4 进程并发的总吞吐为 1.86x / 2.64x / 3.03x，默认取 4。
#
# 用法：
#   PY=/path/to/python bash scripts/run_day4.sh
#   N_SHARDS=3 bash scripts/run_day4.sh
#
# 各分片日志在 logs/day4_shard_K.log；评估结果在 logs/day4_eval_*.log 与 results/*.csv。
# 中断后直接重跑即可：已存在的 checkpoint 会被跳过。

set -u
cd "$(dirname "$0")/.."

PY="${PY:-python}"
N="${N_SHARDS:-4}"
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1

mkdir -p logs
echo "[$(date +%H:%M)] 启动 $N 个训练分片"
for k in $(seq 0 $((N - 1))); do
  (
    "$PY" scripts/sweep_arch.py --part A --shard "$k/$N"
    "$PY" scripts/sweep_bits.py --shard "$k/$N"
    "$PY" scripts/sweep_arch.py --part B --shard "$k/$N"
  ) > "logs/day4_shard_$k.log" 2>&1 &
done
wait

echo "[$(date +%H:%M)] 全部分片训练结束，开始统一评估"
"$PY" scripts/sweep_arch.py --eval-only > logs/day4_eval_arch.log 2>&1
"$PY" scripts/sweep_bits.py --eval-only > logs/day4_eval_bits.log 2>&1
echo "[$(date +%H:%M)] Day 4 完成"
