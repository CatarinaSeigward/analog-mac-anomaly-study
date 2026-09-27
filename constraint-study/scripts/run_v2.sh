#!/usr/bin/env bash
# 仿真器 v2：固定满量程、饱和激活、三种读噪声模型，60 个 HWA 任务（见 scripts/sweep_v2.py）。
# 与 v1.1（run_v11.sh）的同 seed 结果配对比较，所以要在 run_v11.sh 之后跑。
#
# 内存：每个训练进程约 1.4 GB 主机内存，按空闲物理内存定 N_SHARDS。
#
#   PY=/path/to/python N_SHARDS=3 bash scripts/run_v2.sh
set -u
cd "$(dirname "$0")/.."
PY="${PY:-python}"
N="${N_SHARDS:-3}"
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
mkdir -p logs

echo "[$(date +%H:%M)] 启动 $N 个训练分片"
for k in $(seq 0 $((N - 1))); do
  "$PY" scripts/sweep_v2.py --shard "$k/$N" >| "logs/v2_shard_$k.log" 2>&1 &
done
wait

echo "[$(date +%H:%M)] 训练结束，开始统一评估"
"$PY" scripts/sweep_v2.py --eval-only >| logs/v2_eval.log 2>&1
echo "[$(date +%H:%M)] 完成"
