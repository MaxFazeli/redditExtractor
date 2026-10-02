cd /app/sbatch_scheduler

python sbatch_scheduler.py \
  --sbatch-script ollama_serve.sh \
  -u marweb \
  -host marweb@nvcluster.tugraz.at \
  --ollama-port=28010 \
  --port-forwarding \
  --script /app/scripts/bug_insertion.sh \
  --stop

