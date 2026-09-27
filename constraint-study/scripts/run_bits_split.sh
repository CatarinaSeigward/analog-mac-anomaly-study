#!/usr/bin/env bash
# 第 0 步（NEXT-STEPS 0-1）：补 5-bit 点，并把 W / S / B 位宽分开扫，定位 4-bit 悬崖在哪一项。
#
#   W/S/B 同步 5-bit              3 个任务  -> results/bits_r0.04_b5.csv
#   只扫 W / S / B，4 与 5 bit    各 6 个   -> results/bits_r0.04_{w,s,b}_b4-5.csv
#
# 6-bit 参照和同步 4-bit 用已有的 bits_wsb{4,6}_r0.04（图 4），不重训。
# none 架构下 S 位宽只作用于首尾两个转换器，W 是电导电平数，B 是偏置 DAC。
#
# 每条命令各自按轮询分片；分片号逐条错开一位，21 个任务在 4 个进程间按 5/5/6/5 分配。
#
# 内存：每个训练进程约占 1.4 GB 主机内存。进程数超过可提交内存时报
#   "fatal : Memory allocation failure" / CUBLAS_STATUS_EXECUTION_FAILED —— 看着像显存，
#   其实是主机内存。先看空闲物理内存再定 N_SHARDS。
# 中途停止：只结束 bash 不够，子 shell 会接着启动下一条命令。要按命令行把
#   run_bits_split 和 sweep_bits 的进程都结束。
#
#   PY=/path/to/python N_SHARDS=2 bash scripts/run_bits_split.sh
set -u
cd "$(dirname "$0")/.."
PY="${PY:-python}"
N="${N_SHARDS:-4}"
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
mkdir -p logs

CMDS=("--bits 5" "--sweep-only w --bits 4 5" "--sweep-only s --bits 4 5" "--sweep-only b --bits 4 5")

echo "[$(date +%H:%M)] 启动 $N 个训练分片"
for k in $(seq 0 $((N - 1))); do
  (
    for c in "${!CMDS[@]}"; do
      # shellcheck disable=SC2086
      "$PY" scripts/sweep_bits.py ${CMDS[$c]} --shard "$(( (k + c) % N ))/$N"
    done
  ) > "logs/bits_split_shard_$k.log" 2>&1 &
done
wait

echo "[$(date +%H:%M)] 训练结束，开始统一评估"
for c in "${!CMDS[@]}"; do
  # shellcheck disable=SC2086
  "$PY" scripts/sweep_bits.py ${CMDS[$c]} --eval-only
done > logs/bits_split_eval.log 2>&1
echo "[$(date +%H:%M)] 完成"
