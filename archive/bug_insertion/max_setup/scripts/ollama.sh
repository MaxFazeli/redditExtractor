#!/bin/bash

PORT=$1
SCRIPT_PATH=$2
GPUS_PER_NODE=$3

echo "Running on port: $PORT"
echo "Using script: $SCRIPT_PATH"
echo "GPUs per node: $GPUS_PER_NODE"

cd /app/sbatch_scheduler

python sbatch_scheduler.py \
  --sbatch-script ollama_serve.sh \
  -u marweb \
  -host marweb@nvcluster.tugraz.at \
  --ollama-port=$PORT \
  --gpus-per-node $GPUS_PER_NODE \
  --port-forwarding \
  --script $SCRIPT_PATH \
  --stop

