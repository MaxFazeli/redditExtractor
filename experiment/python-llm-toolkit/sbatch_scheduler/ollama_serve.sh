#!/bin/bash
#SBATCH --partition=hcc
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --gpus=1
#SBATCH --job-name=ollama_serve
#SBATCH --output=ollama_serve_%j.log
#SBATCH --error=ollama_serve_%j.err

# Define a variable for the port number (replaced by sbatch_scheduler.py --ollama-port)
PORT_NUMBER=11434

# Record the environment of this run (hardware + Ollama version), useful for reporting
echo "Job $SLURM_JOB_ID on $(hostname) - $(date -Iseconds)"
echo "Ollama: $(~/bin/ollama --version 2>&1 | tail -n 1)"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv 2>&1 || true

# Run the Ollama server (stdout/stderr go to the per-job log files above).
# Bound to 127.0.0.1: only reachable through the SSH tunnel, not by other users on the cluster network.
OLLAMA_HOST=127.0.0.1:$PORT_NUMBER exec ~/bin/ollama serve
