# Running the Recommendation Experiment

How the pipeline works is described in [`docs/experiment-pipeline.md`](../../../docs/experiment-pipeline.md).
This file covers the commands.

## Compose files

| File | What it runs | Results |
|---|---|---|
| `docker-compose.smoke_test.yml` | 2 models (llama3.2:3b, gpt-oss:20b), first 3 submissions, zero- + few-shot, 1 GPU (quick "does it work at all" check) | `setup/results/smoke_test/` |
| `docker-compose.test_all_models.yml` | 12 Ollama models, first 10 submissions, zero-shot (pipeline check) | `setup/results/test_validation/` |
| `docker-compose.full_run.yml` | 12 Ollama models, all 1,718 submissions, zero- + few-shot | `setup/results/full_run/` |
| `docker-compose.openai_run.yml` | 3 GPT models via the API (no cluster), all submissions, zero- + few-shot | `setup/results/full_run/` |
| `docker-compose.mongodb-test.yml` | MongoDB only, on `localhost:27017` (local development) | – |

Each file uses its own service names (`mongodb_test`/`recommendation_test`, `…_full`, `…_openai`), so the
files can run at the same time without name conflicts. A top-level `name:` is not used because the
server's Docker Compose (v2.2.3) does not support it. Two instances that use the cluster need different
Ollama ports (28021 test, 28022 full run).

## 0. One-time prerequisites

**On nvcluster**: Ollama in `~/bin/ollama`. The installed version (checked 2026-10-01) is **0.7.1**,
but the `gpt-oss` models need Ollama ≥ 0.11, so update it first. To install or update:
```bash
ssh hungryapeman@nvcluster.tugraz.at
cd ~ && wget https://ollama.com/download/ollama-linux-amd64.tgz && tar -xzf ollama-linux-amd64.tgz
~/bin/ollama --version
```

**On funkenzutzler**: an SSH key in `~/.ssh` that logs in to `hungryapeman@nvcluster.tugraz.at` without a
password, and to the compute nodes (`nvcluster-nodeN.tugraz.at`). The container mounts `~/.ssh` read-only.
```bash
ssh -o BatchMode=yes hungryapeman@nvcluster.tugraz.at hostname    # must work without a prompt
```

Rootless Docker on funkenzutzler:
```bash
export PATH=$HOME/bin:$PATH
export DOCKER_HOST=unix:///run/user/10024/docker.sock
```

## 1. Copy the code to funkenzutzler

The server holds a copy of the `experiment/` folder in `~/experiment`, plus the dataset in `~/experiment/dataset/`.
From Windows (PowerShell, in `redditExtractor/`):
```powershell
scp -P 4774 -i $HOME\.ssh\TUServer -r experiment\* mfazeli@funkenzutzler.isds.tugraz.at:/home/mfazeli/experiment/
# only once: the dataset
scp -P 4774 -i $HOME\.ssh\TUServer data\backup_mongo_dump\reddit_data.dataset.json mfazeli@funkenzutzler.isds.tugraz.at:/home/mfazeli/experiment/dataset/
```
`scp -r` also copies `setup/results`. Don't copy older local results over newer results on the server.
If needed, exclude that folder, or copy only the files you changed.

## 2. Test run (recommended before the full run)

```bash
cd ~/experiment/setup/docker
docker compose -f docker-compose.test_all_models.yml up -d --build
docker compose -f docker-compose.test_all_models.yml logs -f recommendation_test   # Ctrl+C stops following, not the run
tail -f ../results/test_validation/run.log
```
Check that the summary at the end of `run.log` shows `[OK]` for every model. Every model that did not
finish (including "not available") is marked `[FAIL]` with its exit code.

Starting the same file again resumes. To repeat the test from scratch, change `RUN_NAME` or move the
folder `../results/test_validation` away.

## 3. Full run

```bash
docker compose -f docker-compose.full_run.yml up -d --build
tail -f ../results/full_run/run.log

# GPT models (can run in parallel, no GPU needed). OPENAI_API_KEY is read from setup/docker/.env
docker compose -f docker-compose.openai_run.yml up -d --build
```

**Expected duration (Ollama):** based on the 10-submission test, about 4–5 days per prompt type for
all 12 models on 2 GPUs. Few-shot prompts are longer, so expect more for few-shot. The `hcc` partition
has no time limit. If the run stops (network, server reboot, ...), start it again with the same command: finished models
and already answered submissions are skipped.

**Stop:** `docker compose -f docker-compose.full_run.yml down` stops the container, and the scheduler
cancels the SLURM job (`scancel`) before exiting. If a container was killed hard, check the cluster
for leftover jobs: `ssh hungryapeman@nvcluster.tugraz.at squeue -u hungryapeman`, and `scancel <id>`
if needed.

## 4. Evaluate

```bash
docker compose -f docker-compose.full_run.yml run --rm --no-deps recommendation_full \
  python recommendation_evaluator.py --results-dir /app/results/full_run
```
Output goes to `setup/results/full_run/evaluated/` (per-row metrics, aggregates, format adherence).
The evaluator can also run anywhere with `pandas` installed: `python recommendation_evaluator.py --results-dir <dir>`.

## 5. Troubleshooting

| Symptom | Check |
|---|---|
| `Job ... is not in the queue anymore` | SLURM job failed at start: `ollama_serve_<jobid>.err` in the cluster home directory |
| `Ollama did not become reachable` | Is the port already used by someone else on the node? Use another port in the compose `command:` |
| Model `NOT AVAILABLE` (exit 2) | Model name wrong, or Ollama on the cluster too old for it (`~/bin/ollama --version`, update as in §0). The server version is printed in the log and stored in `progress_<model>.json` |
| `INFRASTRUCTURE ERROR` (exit 3) | Tunnel or job lost (time limit?). Start the same compose file again to resume |
| SSH asks for a password / `Permission denied (publickey)` | Key setup in §0. Inside the container, ssh runs with `BatchMode=yes` and fails instead of prompting |
| MongoDB container unhealthy | `docker compose ... logs mongodb_full` (or `_test`): is the dataset file at `experiment/dataset/reddit_data.dataset.json`? (or set `DATASET_FILE`) |

## 6. Local development (Windows/Linux, no cluster)

```bash
# MongoDB with the dataset on localhost:27017
DATASET_FILE=../../../data/backup_mongo_dump/reddit_data.dataset.json \
  docker compose -f docker-compose.mongodb-test.yml up -d

# offline unit tests (no MongoDB/LLM needed)
cd ../.. && python -m unittest tests/test_units.py -v

# real run against a local Ollama (OLLAMA_HOST, default http://localhost:11434)
python recommendation_experiment.py --model llama3.2:3b-instruct-q4_K_M --limit 5 --output-dir test_results
```

## 7. Server facts (checked 2026-10-01)

| Item | Value |
|---|---|
| funkenzutzler | `ssh -i ~/.ssh/TUServer -p 4774 mfazeli@funkenzutzler.isds.tugraz.at` (TU Graz VPN); experiment in `~/experiment` |
| Docker | rootless, `~/bin/docker`, `DOCKER_HOST=unix:///run/user/10024/docker.sock`; Docker Compose **v2.2.3** |
| funkenzutzler → nvcluster | key `~/.ssh/id_rsa` → `hungryapeman@nvcluster.tugraz.at` works without password |
| nvcluster Ollama | `~/bin/ollama` **0.7.1** (too old for gpt-oss) |
| Partition `hcc` | 2 nodes × 8 NVIDIA Quadro RTX 8000 (48 GB each), time limit: infinite |
| `setup/docker/.env` | contains `OPENAI_API_KEY` (the line starts with a space – remove it if Compose does not pick up the key) |
