#!/usr/bin/env bash
set -euo pipefail

GPUS=${GPUS:-4}
CONFIG=${CONFIG:-configs/experiments/bert_finetune.yaml}
MASTER_PORT=${MASTER_PORT:-29500}

echo "Launching distributed training on $GPUS GPUs"
echo "Config: $CONFIG"

torchrun \
  --nproc_per_node=$GPUS \
  --master_port=$MASTER_PORT \
  scripts/train.py \
  --config "$CONFIG" \
  "$@"
