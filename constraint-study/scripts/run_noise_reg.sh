#!/usr/bin/env bash
# NEXT-STEPS 0-2：噪声正则化的全 fp32 对照，21 个任务（7 组 × 3 seeds），见 scripts/probe_noise_reg.py。
#
# 内存：每个训练进程约 1.4 GB 主机内存，按空闲物理内存定 N_SHARDS。
#
#   PY=/path/to/python N_SHARDS=3 bash scripts/run_noise_reg.sh
set -u
cd "$(dirname "$0")/.."
PY="${PY:-python}"
N="${N_SHARDS:-2}"
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
mkdir -p logs

echo "[$(date +%H:%M)] 启动 $N 个训练分片"
for k in $(seq 0 $((N - 1))); do
  "$PY" scripts/probe_noise_reg.py --shard "$k/$N" >| "logs/noise_reg_shard_$k.log" 2>&1 &
done
wait

echo "[$(date +%H:%M)] 训练结束，开始统一评估"
"$PY" scripts/probe_noise_reg.py --eval-only >| logs/noise_reg_eval.log 2>&1
echo "[$(date +%H:%M)] 完成"
