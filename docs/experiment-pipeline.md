# Experiment Pipeline: LLM Recommendations on nvcluster

This document describes how the recommendation experiment is executed: where the data comes from,
how the LLMs are reached, how scripts are started, and where results are stored. It reflects the
state after the clean-up of 2026-10-01; all changes made then are listed with their rationale in
[changelog.md](changelog.md).

## 1. Goal

For each of the 1,718 r/ifyoulikeblank submissions in the final dataset, ask several LLMs to recommend
items (books, films/TV shows, music, games) and store the answers together with the recommendations
that human commenters gave (ground truth). A separate evaluation step compares the two.

- **Models:** 20 open-weight models served by Ollama on the nvcluster GPU cluster, and 3 OpenAI
  models called through the API (Table 1).
- **Prompting:** zero-shot, and few-shot with 3 examples from the same domain.
- **Input:** the submission title plus its body text (`selftext`), as seen by the human commenters. A switch restricts it to the title.

## 2. Components

| Component | Where it runs | Role |
|---|---|---|
| **funkenzutzler** (`funkenzutzler.isds.tugraz.at`) | institute server | Runs the Docker containers (rootless Docker). Has SSH access to nvcluster. |
| **MongoDB container** | funkenzutzler | Imports `reddit_data.dataset.json` into `reddit_data.dataset` at start-up. Read-only source of the submissions. |
| **Experiment container** | funkenzutzler | Python 3.13 image with the experiment dependencies. The `experiment/` folder is mounted, not copied, so code changes need no rebuild. Mounts `~/.ssh` read-only for the cluster connection. |
| **sbatch_scheduler.py** (`python-llm-toolkit/`) | experiment container | Orchestrator: submits the SLURM job, waits for it, opens the SSH tunnel, runs the experiment, cancels the job. |
| **nvcluster** (`nvcluster.tugraz.at`) | TU Graz GPU cluster | SLURM login node. The job runs on a GPU compute node (`nvcluster-node2`, `nvcluster-node4`) in partition `hcc`: 2 nodes with 8× NVIDIA Quadro RTX 8000 (48 GB) each, no time limit. The experiment requests 2 GPUs. |
| **Ollama server** (`~/bin/ollama serve`) | GPU compute node | Serves the open-weight models. Bound to `127.0.0.1:<port>` on the node. |
| **OpenAI API** | external | Serves the GPT models. Called directly from the container, no cluster involved. |

## 3. Execution flow

```
funkenzutzler: docker compose -f docker-compose.full_run.yml up -d
│
├─ mongodb container
│    mongoimport --drop  dataset.json → reddit_data.dataset      (healthcheck: import finished)
│
└─ experiment container  (starts once MongoDB is healthy)
     setup/scripts/ollama.sh <port> setup/scripts/run_models.py <gpus>
     └─ exec python sbatch_scheduler.py ...          (all cluster commands via ssh hungryapeman@nvcluster)
          1. fill in ollama_serve.sh template (port, partition, GPUs)
          2. upload it as ~/ollama_serve_<port>.sh, then: sbatch ~/ollama_serve_<port>.sh → job id
          3. poll squeue every 10 s until the job state is RUNNING → node name (e.g. nvcluster-node2)
             (gives up if the job ends or disappears from the queue)
          4. SSH tunnel  container:11434 → nvcluster-node2.tugraz.at:127.0.0.1:<port>
             (background thread; restarted automatically if it drops)
          5. wait until GET http://localhost:11434/api/version answers (max 10 min)
          6. run run_models.py  ──────────────────────────────────────────────┐
          7. close the tunnel, scancel <job id>     (also on errors and docker stop)
                                                                              │
run_models.py: for each model (sequentially, one subprocess per model) ◄──────┘
   recommendation_experiment.py --model M --progress-file results/<RUN>/progress_<M>.json
     • load all submissions from MongoDB, ordered by _id
     • check the model is available (Ollama: show/pull; OpenAI: models.retrieve)
     • for each prompt type, for each submission not yet in the result CSV:
          build prompt → call LLM → parse JSON → append one row to the CSV (flushed)
```

### 3.1 SLURM job (`ollama_serve.sh`)
```bash
#SBATCH --partition=hcc  --nodes=1  --gpus-per-node=<n>  --gpus=<n>
#SBATCH --output=ollama_serve_%j.log  --error=ollama_serve_%j.err
echo Ollama version; nvidia-smi --query-gpu=...          # recorded for the documentation
OLLAMA_HOST=127.0.0.1:<port> exec ~/bin/ollama serve
```
The job runs until the scheduler cancels it, or until the partition's time limit ends it. The logs
`ollama_serve_<jobid>.log/.err` in the cluster home directory record the GPU type, driver and Ollama version.

### 3.2 Connection to the cluster
- Authentication uses the SSH key in `~/.ssh` on funkenzutzler, mounted read-only into the container.
  Every SSH call uses `BatchMode=yes`, so it fails fast instead of waiting for a password prompt.
- The Ollama port must be unique per running instance, because other users share the GPU nodes:
  28021 for the test run, 28022 for the full run.
- The Ollama API is reachable only through the tunnel. On the node it listens on `127.0.0.1`.

## 4. Experiment logic (`recommendation_experiment.py`)

### 4.1 Prompt construction
- **Prompt template** per domain and prompt type: `prompts/{zero,few}_shot_prompts/<domain>_<type>.txt`.
  Film and TV share the `movie_tv` template. The placeholder `<submission_text>` receives the request.
- **Request text** (`--input-mode`): `title_selftext` (default: title, a blank line, then the body if present; `[deleted]`/`[removed]` bodies are ignored) or `title` (title only).
- **Few-shot examples:** 3 other submissions of the same domain, each with up to 10 of its
  ground-truth recommendations. They are drawn with a random generator seeded with `seed:submission_id`, so:
  - every model gets the same examples for the same submission;
  - the selection is reproducible and does not depend on processing order or `--limit`;
  - the chosen example ids are stored in the result row (`few_shot_example_ids`).
- The whole prompt is sent as the user message, with no system prompt.

### 4.2 Generation settings

| Setting | Ollama models | OpenAI models |
|---|---|---|
| Output format | `format="json"` | `response_format={"type":"json_object"}` |
| Temperature / top-p | model defaults | API default (GPT-5 models accept only the default) |
| Seed | 42 (`config.SEED`) | not sent |
| Max output tokens | 4096 (`num_predict`; includes the "thinking" of reasoning models) | API default |
| Context window | 8192 (`num_ctx`) | model default |
| CPU threads | 2 (`num_thread`) | – |

All settings, the Ollama server version and the input mode are stored in `progress_<model>.json → setup.runConfig`.

### 4.3 Response parsing
- `valid_json`: the response parses as JSON (after stripping markdown code fences).
- `valid_structure`: the response has the requested structure `{"recommendations": [...]}` (or is a bare list).
  Deviating structures such as `{"books": [...]}` are still parsed leniently, but flagged.
- `nr_recs` / `nr_recs_unique`: number of recommendations / distinct titles (case-insensitive).
- `response_truncated`: generation hit the token limit (`finish_reason == "length"`).
- `prompt_truncated`: the prompt used ≥ 95 % of the context window, so Ollama may have cut it.

### 4.4 Error handling

| Situation | Handling | Effect on results |
|---|---|---|
| Model cannot be pulled / unknown API model | model marked `not available`, next model continues | exit code 2, shown as FAIL in `run.log` |
| Connection error, timeout, server error | retried 3× with 30/60 s back-off | none if a retry succeeds |
| Error persists after retries | run stops, model marked `failed`, **row not written** | exit code 3; resume later |
| Model answers with invalid JSON / wrong structure | row written with `valid_json` / `valid_structure` = False | part of the results (format adherence) |
| Container stopped (`docker compose down`) | scheduler receives SIGTERM, cancels the SLURM job | current submission is redone on resume |

## 5. Results

### 5.1 Layout
```
experiment/setup/results/            (host)  =  /app/results  (container)
└── <RUN_NAME>/                      e.g. test_validation, full_run
    ├── run.log                      start/end/status of every model
    ├── progress_<model>.json        status per model and prompt type + run configuration
    └── <model>/
        ├── zero_shot_results.csv
        └── few_shot_results.csv
```
The folder is a bind mount, so results survive container restarts and removal.
Experiment logs (loguru) are written to `experiment/log/`.

### 5.2 Result CSV columns

| Column | Meaning |
|---|---|
| `submission_id`, `domain`, `model`, `prompt_type`, `input_mode` | identification |
| `submission_text` | request text sent to the model |
| `few_shot_example_ids` | ids of the example submissions (few-shot only) |
| `valid_json`, `valid_structure` | format adherence (§4.3) |
| `nr_recs`, `nr_recs_unique`, `nr_ground_truth` | counts |
| `llm_recommendations`, `ground_truth` | JSON arrays |
| `raw_response` | unmodified model output |
| `finish_reason`, `response_truncated`, `prompt_truncated` | generation status |
| `prompt_tokens`, `response_tokens`, `duration_seconds` | cost / latency |
| `error`, `timestamp` | parse error message; time of the call |

### 5.3 Resuming
Each row is written and flushed as soon as it is produced. `progress_<model>.json` is written atomically
(temporary file + rename). Starting the same `RUN_NAME` again:
- skips models whose prompt types are all `finished`;
- skips submissions already in the CSV;
- reuses the input mode and seed stored in the progress file, so one run never mixes settings.

## 6. Evaluation (`recommendation_evaluator.py`)
```
python recommendation_evaluator.py --results-dir setup/results/full_run
```
- Matching: ID (IMDb / ISBN) → exact title → fuzzy title (SequenceMatcher ≥ 0.9).
- First 10 unique recommendations; precision, recall, F1, nDCG, each with exact and fuzzy matching.
- Rows with invalid JSON or fewer than 5 ground-truth items are excluded from the metrics. Format
  adherence is reported separately in `format_adherence_by_model_prompttype.csv`.
- If a model/prompt/submission occurs in several runs under `--results-dir`, only the most recent row is used.
- The output folder holds a `.evaluation_output` marker and is never re-read as input.

## 7. Models (Table 1)

| Size class | Ollama (q4_K_M unless stated) | OpenAI |
|---|---|---|
| tiny (< 4B) | llama3.2:3b-instruct | gpt-5-nano |
| small (4–10B) | gemma3:4b-it, llama3.1:8b-instruct, deepseek-r1:8b-0528-qwen3, **qwen3.5:4b**, **gemma4:e4b-it**, **olmo-3:7b-instruct**, **qwen3.5:9b** | – |
| medium (10–50B) | gpt-oss:20b, mistral-small3.2:24b-instruct-2506, gemma3:27b-it, qwen3:32b, deepseek-r1:32b-qwen-distill, **qwen3.5:27b**, **gemma4:31b-it**, **olmo-3.1:32b-instruct** | gpt-5-mini |
| large (≥ 50B) | llama3.3:70b-instruct, gpt-oss:120b, mistral-large:123b-instruct-2411, **qwen3.5:122b-a10b** (MoE) | gpt-5.2 |

**Bold** = added on 2026-10-02 (`generation: 2` in models.py): qwen3.5 covers every size class (scaling within one family), gemma4 is the successor of gemma3 (generation comparison), olmo-3 is fully open incl. training data (relevant for discussing whether Reddit data was seen during training). All Ollama models use q4_K_M quantisation. Claude models (Haiku 4.5 / Sonnet 5.5 / Opus 5.5) are planned as a second proprietary family, to be added together with the API runs.

The size class of the OpenAI models is the tier they are compared against; their parameter counts are
not published. The single source of truth is `experiment/framework/models.py`. Setups:
`tiny_models`, `small_models`, `medium_models`, `large_models`, `tiny_and_small_models` (fit on one GPU), `ollama_models`, `openai_models`, `all_models`.

## 8. Configuration reference

| What | Where |
|---|---|
| Run name, models, prompt types, limit, input mode | `environment:` in the compose file (`RUN_NAME`, `MODEL_SETUP`/`MODELS`, `PROMPT_TYPES`, `LIMIT`, `INPUT_MODE`) |
| Ollama port, number of GPUs | `command:` in the compose file (`ollama.sh <port> <script> <gpus>`) |
| Cluster user / host / partition | env `NVCLUSTER_USER`, `NVCLUSTER_HOST`, `SLURM_PARTITION` (defaults: hungryapeman, nvcluster.tugraz.at, hcc) |
| Generation options, seed, retries, timeouts | `experiment/framework/config.py` |
| Models | `experiment/framework/models.py` |
| Dataset file | env `DATASET_FILE` (default `experiment/dataset/reddit_data.dataset.json`) |

How to run it step by step: [`experiment/setup/docker/README.md`](../experiment/setup/docker/README.md).
