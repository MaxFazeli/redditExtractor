#!/bin/bash
# Container entry point: start Ollama on nvcluster (SLURM), open the SSH tunnel and run a script.
#
# Usage: ollama.sh <ollama-port> <script> <gpus-per-node>
#   ollama-port    port Ollama listens on at the GPU node (1024-65535, unique per running instance)
#   script         .py or .sh script to run once Ollama is reachable on localhost:11434
#   gpus-per-node  number of GPUs to request
#
# Environment:
#   NVCLUSTER_USER   cluster user name           (default: hungryapeman)
#   NVCLUSTER_HOST   cluster login node          (default: nvcluster.tugraz.at)
#   SLURM_PARTITION  SLURM partition             (default: hcc)
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <ollama-port> <script> <gpus-per-node>" >&2
  exit 1
fi

PORT=$1
SCRIPT_PATH=$2
GPUS_PER_NODE=$3

NVCLUSTER_USER=${NVCLUSTER_USER:-hungryapeman}
NVCLUSTER_HOST=${NVCLUSTER_HOST:-nvcluster.tugraz.at}
SLURM_PARTITION=${SLURM_PARTITION:-hcc}

if ! [[ "$PORT" =~ ^[0-9]+$ ]] || [ "$PORT" -lt 1024 ] || [ "$PORT" -gt 65535 ]; then
  echo "Invalid port '$PORT' (must be 1024-65535)" >&2
  exit 1
fi
if [ ! -f "$SCRIPT_PATH" ]; then
  echo "Script not found: $SCRIPT_PATH" >&2
  exit 1
fi

echo "Running on port: $PORT"
echo "Using script: $SCRIPT_PATH"
echo "GPUs per node: $GPUS_PER_NODE"
echo "Cluster: $NVCLUSTER_USER@$NVCLUSTER_HOST (partition $SLURM_PARTITION)"

cd /app/experiment/python-llm-toolkit/sbatch_scheduler

# exec: the scheduler replaces this shell, so `docker stop` (SIGTERM) reaches it directly and it
# can cancel the SLURM job before the container exits.
exec python sbatch_scheduler.py \
  --sbatch-script ollama_serve.sh \
  -u "$NVCLUSTER_USER" \
  -host "$NVCLUSTER_USER@$NVCLUSTER_HOST" \
  --partition "$SLURM_PARTITION" \
  --ollama-port="$PORT" \
  --gpus-per-node "$GPUS_PER_NODE" \
  --port-forwarding \
  --script "$SCRIPT_PATH" \
  --stop
