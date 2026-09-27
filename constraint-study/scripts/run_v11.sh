#!/usr/bin/env bash
# v1.1 = v1 的物理模型 + STE 修复（2026-09-22）：重跑报告里用量化训练得到的全部结果。
#
#   图 3  sweep_arch.py   Part A + B，54 个 HWA 任务   -> results/arch_stefix.csv
#   图 2  sweep_noise.py  C3 的 24 个 HWA 任务         -> results/noise_stefix.csv
#         （C1 / C2 是 fp32 训练，不经过量化器，不受 STE 修复影响，沿用原 checkpoint；C4 复用 C3）
#   §4.2  probe_bn_calib.py                            -> results/bn_calib_stefix.csv
#   §3.3  probe_train_noise.py                         -> results/train_noise_cross_stefix.csv
#
# 物理模型保持 v1（动态量程、ReLU、按比例读噪声）——只换代码版本，结果变了就只能归因于 STE 修复。
# v2 的物理改动（固定满量程、饱和激活、噪声模型）另行重跑，不混在这里。
#
# 内存：每个训练进程约 1.4 GB 主机内存，按空闲物理内存定 N_SHARDS。
#
#   PY=/path/to/python N_SHARDS=3 bash scripts/run_v11.sh
set -u
cd "$(dirname "$0")/.."
PY="${PY:-python}"
N="${N_SHARDS:-3}"
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
mkdir -p logs

echo "[$(date +%H:%M)] 启动 $N 个训练分片"
for k in $(seq 0 $((N - 1))); do
  (
    "$PY" scripts/sweep_arch.py --tag stefix --shard "$k/$N"
    "$PY" scripts/sweep_noise.py --tag stefix --seeds 0 1 2 3 --shard "$(( (k + 1) % N ))/$N"
  ) >| "logs/v11_shard_$k.log" 2>&1 &
done
wait

echo "[$(date +%H:%M)] 训练结束，开始统一评估"
"$PY" scripts/sweep_arch.py --tag stefix --eval-only >| logs/v11_eval_arch.log 2>&1
"$PY" scripts/sweep_noise.py --tag stefix --seeds 0 1 2 3 >| logs/v11_eval_noise.log 2>&1
"$PY" scripts/probe_bn_calib.py --tag stefix >| logs/v11_bn_calib.log 2>&1
"$PY" scripts/probe_train_noise.py --tag stefix >| logs/v11_train_noise.log 2>&1
echo "[$(date +%H:%M)] 完成"
