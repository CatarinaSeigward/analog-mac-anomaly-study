#!/usr/bin/env bash
# 图 4 用 STE 修复后的代码补齐（2026-09-22）。run_ste_check.sh 已训好 W/S/B 同步 4/5/6 与只降 W 的 4/5；
# 这里补剩下的 21 个任务：
#
#   W/S/B 同步 2 / 8 / 10 bit   9 个
#   只降 S，4 / 5 bit           6 个
#   只降 B，4 / 5 bit           6 个
#
# 训练完统一评估。W/S/B 同步的 6 个位宽放在同一次评估里，结果写进一个文件：
#   results/bits_r0.04_b2-4-5-6-8-10_stefix.csv   以及 _s_ / _b_ / _w_ 的单项文件
# （同步 4/5/6 会被重评一次；评估的 C2C 噪声不固定种子，数字可能有千分位的变动。）
#
# 内存：每个训练进程约 1.4 GB 主机内存，按空闲物理内存定 N_SHARDS。
#
#   PY=/path/to/python N_SHARDS=3 bash scripts/run_fig4_stefix.sh
set -u
cd "$(dirname "$0")/.."
PY="${PY:-python}"
N="${N_SHARDS:-2}"
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
mkdir -p logs

TRAIN=("--bits 2 8 10 --tag stefix" "--sweep-only s --bits 4 5 --tag stefix"
       "--sweep-only b --bits 4 5 --tag stefix")
EVAL=("--bits 2 4 5 6 8 10 --tag stefix" "--sweep-only w --bits 4 5 --tag stefix"
      "--sweep-only s --bits 4 5 --tag stefix" "--sweep-only b --bits 4 5 --tag stefix")

echo "[$(date +%H:%M)] 启动 $N 个训练分片"
for k in $(seq 0 $((N - 1))); do
  (
    for c in "${!TRAIN[@]}"; do
      # shellcheck disable=SC2086
      "$PY" scripts/sweep_bits.py ${TRAIN[$c]} --shard "$(( (k + c) % N ))/$N"
    done
  ) >| "logs/fig4_stefix_shard_$k.log" 2>&1 &
done
wait

echo "[$(date +%H:%M)] 训练结束，开始统一评估"
for c in "${!EVAL[@]}"; do
  # shellcheck disable=SC2086
  "$PY" scripts/sweep_bits.py ${EVAL[$c]} --eval-only
done >| logs/fig4_stefix_eval.log 2>&1
echo "[$(date +%H:%M)] 完成"
