# Changelog: Experiment Pipeline Clean-up (2026-10-01)

This log records every change made to the experiment pipeline on 2026-10-01: what was wrong, what was
changed, and why. Entries marked **[M]** affect the experimental method (what the models receive,
or how outputs are generated or scored) and should be mentioned in the methods chapter. The others
concern infrastructure and robustness.

The current architecture is described in [experiment-pipeline.md](experiment-pipeline.md).
Archived files are listed in [`archive/README.md`](../archive/README.md).

---

## 0. Starting point

The pipeline had been adapted from a colleague's "bug insertion" project, which runs LLMs over Python
code via Ollama on nvcluster. Two test runs on 2026-05-28 and 2026-05-29 (10 submissions × 12
Ollama models, zero-shot) had completed. Analysing the code and these results showed:

- the run summary reported **12/12 OK**, but the two `gpt-oss` models had **never run** (`"not available"` in `progress.json`);
- the GPT models in the default model list could never have run (wrong routing, see C2);
- results were only saved after a model had processed all submissions (a crash lost everything);
- several orchestration issues could leave GPU jobs running, or let the run hang forever.

## 1. Decisions taken

| Decision | Choice | Reason |
|---|---|---|
| Request text given to the model **[M]** | **Title + body** (`selftext`, appended below the title if present; `[deleted]`/`[removed]` ignored). Title only remains available via `INPUT_MODE=title`. *Initially "title only" was chosen to stay consistent with the May test runs; changed the same day after the smoke tests.* | The human commenters who produced the ground truth saw title and body, so the models get the same information. 1,240 of 1,718 submissions (72 %) have a body, often with the actual wish (e.g. "Dark humor, decadence, frenetic energy… these seem to be my prerequisites"). Runs up to and including both smoke tests used title only. |
| Sampling **[M]** | **Model-default temperature / top-p, fixed seed (42)** | Keeps each model's intended decoding behaviour while making runs repeatable on the same hardware and Ollama version. GPT-5 models only accept the default temperature anyway. |
| Output-token limit **[M]** | **4096 tokens for all models** (was 1000). Reasoning effort left at model defaults | Smoke test: gpt-oss:20b spent the entire 1000-token budget on internal reasoning and returned empty or cut-off JSON in 6 of 6 cases (§8). One uniform, higher limit keeps the comparison fair; non-reasoning models stop on their own (≤ ~620 tokens) and are unaffected. Remaining truncations are visible in `response_truncated`. |

---

## 2. Changes

### A. Clean-up

| ID | Change | Rationale |
|---|---|---|
| A1 | Bug-insertion scripts (`tiny/small/medium/large_models*.sh`, `bug_insertion.sh`, `ollama_simple.sh`) and the bug-insertion `docker-compose.yml` moved to `archive/bug_insertion/` | They call `02_bug_insertion.py`, which is not part of this project. |
| A2 | Bug-insertion constants removed from `framework/config.py`; bug-insertion JSON schema validation removed from `framework/json_utils.py`; unused `framework/utils.py` archived. Originals kept in `archive/bug_insertion/framework/` | Dead code that suggested a different experiment. |
| A3 | `recommendation_experiment_lite.py`, its test prompt, `run_experiment.sh`, `run_test_all_models.py` and `docker-compose.recommendation.yml` moved to `archive/superseded/` | Replaced by `run_models.py` and the new compose files. The lite script was broken for GPT (`openai_client.chat()` does not exist). The old compose file used port 71195. |
| A4 | Outdated docs (`CONTAINER_ORCHESTRATION_DOCUMENTATION.md`, `EXPERIMENT_SETUP_GUIDE.md`, `READY_TO_RUN.md`) archived; replaced by `experiment-pipeline.md` and a rewritten `README.md` | They described the bug-insertion setup. |
| A5 | Stale evaluation folders (`evaluated/`, `evaluated_new/` with nested `evaluated/evaluated/...`) and the duplicate `test_validation_new/` moved to `archive/results_stale/` | Produced by the evaluator bug E1. They mix two runs per model. Raw test results were left in place. |
| A6 | `python-llm-toolkit` (the colleague's git repository) kept as an embedded dependency. Only `sbatch_scheduler/` was modified (section B). `ollama_test/` and `system_prompt_generator/` were left untouched | Keeps the upstream repository intact. Changes show up as `git diff` there. |
| A7 | Top-level folder `max/` (the colleague's full bug-insertion setup: compose files, `prompts.json` with bug-insertion prompts) moved to `archive/bug_insertion/max_setup/`. `README.md` updated (it described a `python_scripts_markus/` folder that no longer exists) | Not used by this experiment. |

### B. Orchestration (`python-llm-toolkit/sbatch_scheduler/`, `setup/scripts/ollama.sh`)

| ID | Problem | Change |
|---|---|---|
| B1 | Job state was taken from the TIME column (`!= "0:00"`) via fixed column positions of the default `squeue` output | `squeue -h -o "%i\|%u\|%T\|%M\|%N"`: explicit fields, and the job counts as started only in state `RUNNING`. |
| B2 | If the job failed before starting, monitoring printed "Job not found, retrying" **forever** | Monitoring fails after 6 consecutive "not found" results, or on a terminal state (`FAILED`, `TIMEOUT`, `CANCELLED`, ...). The job is then cancelled. |
| B3 | The experiment started as soon as the SSH tunnel thread was launched, before Ollama was necessarily reachable | Wait until `GET /api/version` answers through the tunnel (max. 10 min; aborts if the job ends). |
| B4 | A dropped SSH tunnel was never restored | The tunnel thread restarts ssh after 10 s while the job runs. `ExitOnForwardFailure`, `ServerAliveCountMax` added. |
| B5 | `docker stop` sends SIGTERM, but only Ctrl+C was handled. Also, `ollama.sh` ran the scheduler as a bash child, so the signal never reached Python. **The SLURM job (GPUs) kept running after the container was stopped.** | SIGTERM/SIGHUP handled like Ctrl+C. `ollama.sh` uses `exec`. `stop_grace_period: 2m` in the compose files. Verified: `docker stop` → `scancel`. |
| B6 | The script's exit code was ignored. The scheduler always ended "successfully" | The scheduler exits with the script's exit code (1 = scheduler error, 130 = interrupted). `docker compose ps` shows the real outcome. |
| B7 | Every instance wrote `~/ollama_serve.sh` and the same log files on the cluster, so parallel instances interfered | Remote script `~/ollama_serve_<port>.sh`. Logs `ollama_serve_<jobid>.log/.err` via SLURM `%j`. |
| B8 | Port 71195 used in two compose files: invalid (> 65535), Ollama cannot bind | `--ollama-port` validated to 1024–65535 (scheduler and `ollama.sh`). Compose files use 28021 (test) and 28022 (full run). |
| B9 | Ollama listened on `0.0.0.0` on the GPU node, so other cluster users could use it | Bound to `127.0.0.1`. It is only reachable through the tunnel, which connects to `127.0.0.1` on the node anyway. |
| B10 | Hardware and software environment of a run was not recorded | The SLURM log now records the job id, node, Ollama version and `nvidia-smi` (GPU model, memory, driver). |
| B11 | SSH could hang waiting for a password prompt inside the container; no timeouts | All ssh calls use `BatchMode=yes`, `ConnectTimeout=30`; subprocess timeouts. |
| B12 | `check_squeue.py` crashed when run standalone (unpacked 3 of 4 return values) | Returns a dict; CLI fixed. |
| B13 | Cluster user hard-coded in `ollama.sh` | `NVCLUSTER_USER`, `NVCLUSTER_HOST`, `SLURM_PARTITION` env vars (same defaults). Arguments validated. |

### C. Model handling

| ID | Problem | Change |
|---|---|---|
| C1 | `framework/models.py` still listed the colleague's coding models (starcoder, codellama, ...). `--model-setup` therefore selected models that are not part of the thesis | Registry replaced by the 12 Ollama + 3 OpenAI thesis models with size class (Table 1 in experiment-pipeline.md). Setups: `tiny/small/medium/large_models`, `ollama_models`, `openai_models`, `all_models`. The model list exists only in this file now. Previously it was duplicated in three places. |
| C2 **[M]** | GPT models were listed as `'GPT-5 nano'`, `'GPT-5-mini'`, `'GPT-5.2'`. Routing required the prefix `openai`, so they were sent to Ollama and would have been "not available" | Keys `openai:gpt-5-nano`, `openai:gpt-5-mini`, `openai:gpt-5.2`. API ids are taken after the colon. Routing by registry. Availability is checked with `models.retrieve` before the run. |
| C3 | "Model not available" (e.g. gpt-oss) still ended with exit code 0, so the run summary showed OK | Exit code 2. Shown as `FAIL (exit 2) NOT AVAILABLE` in `run.log`. |
| C4 | The gpt-oss failure gave no reason | The pull error is logged with the Ollama server version. Cause confirmed on 2026-10-01: nvcluster runs Ollama **0.7.1**, while gpt-oss needs ≥ 0.11. **To do:** update `~/bin/ollama` (docker README §0). |
| C5 | Large model downloads used one blocking HTTP request (risk of time-out for 70–123B models) | Streaming pull with progress output. HTTP timeout of 30 min per request. |

### D. Experiment script (`recommendation_experiment.py`, `framework/`)

| ID | Problem | Change |
|---|---|---|
| D1 | Results were written only after **all** submissions of a model/prompt type: a crash, SLURM time-out or `docker stop` lost up to ~2 days of computation | `ResultWriter` appends and fsyncs **every row** immediately. |
| D2 | No real resume: a restart re-ran the whole model/prompt type | Rows already in the CSV are skipped. `run_models.py` uses fixed paths `results/<RUN_NAME>/progress_<model>.json`, so restarting the same compose file resumes. A resume reuses the input mode and seed stored in the progress file. |
| D3 | CSV columns were taken from the first row. If the first submission failed, all later rows raised an error, and the whole file was lost | Fixed column list (`RESULT_FIELDNAMES`) for all rows. |
| D4 | Connection errors / time-outs were written as normal result rows with an error message (and could fill a whole run if the tunnel died) | Infrastructure errors are retried 3× (30 s / 60 s back-off). If they persist, the run stops with exit code 3 and **no row is written**, so the submission is redone on resume. Model errors (invalid JSON etc.) are still recorded as results. |
| D5 | `progress.json` was rewritten in place, so a crash during the write could corrupt it | Atomic write (temporary file + `os.replace`). New statuses `failed`, `not available`. |
| D6 **[M]** | Few-shot examples were drawn with unseeded `random.sample`: not reproducible, and **each model saw different examples** for the same submission | Examples are drawn with an RNG seeded with `seed:submission_id` from the domain pool sorted by id. Every model gets identical examples; ids stored in `few_shot_example_ids`. Also faster: the pool is loaded once instead of one MongoDB query per submission. |
| D7 **[M]** | No seed for Ollama, so outputs were not repeatable | `seed=42` sent with every Ollama request. Temperature stays at model default (decision §1). |
| D8 **[M]** | Context window 4096 tokens. Ollama silently drops the **beginning** of longer prompts (the instructions) | `num_ctx` 8192. Rows where the prompt uses ≥ 95 % of the window are flagged `prompt_truncated`. |
| D9 **[M]** | Submission order (and therefore which submissions `--limit N` selects) depended on MongoDB's natural order | Submissions sorted by `_id`. |
| D10 **[M]** | Parsing: a dict without a `recommendations` key was treated as a single recommendation, and a non-string title (e.g. `1984`) raised an exception that turned a valid response into an error row | Lenient extraction of a single list value (e.g. `{"books": [...]}`), flagged with the new column `valid_structure`. Titles converted with `str()`. Non-dict list items ignored. |
| D11 | Truncated outputs (token limit) were not visible | `finish_reason`, `response_truncated` columns. |
| D12 | Inference cost/latency not recorded | `duration_seconds` per row (Ollama: server-side `total_duration`; OpenAI: wall clock). Token counts kept. |
| D13 **[M]** | OpenAI: no JSON mode (Ollama used `format="json"`); `None` content (e.g. refusal) crashed the parser; exceptions lost their message | `response_format={"type": "json_object"}` for parity with Ollama. Empty content handled. The error message is stored in `error`. SDK retries (5×) for 429/5xx. |
| D14 | Run configuration not stored with the results | `progress_<model>.json → setup.runConfig`: input mode, seed, all Ollama options, Ollama server version, few-shot settings. |
| D15 | Paths relative to the current directory (prompts, framework import). `time.tzset()` crashed on Windows | Paths relative to the script. `tzset` only where available. `LOG_DIR` configurable. |
| D16 | Prompt templates were read from disk for every submission | Cached. |

### E. Evaluation (`recommendation_evaluator.py`)

| ID | Problem | Change |
|---|---|---|
| E1 | Re-running the evaluator with another output folder re-read earlier outputs (`*_results.csv` inside `evaluated/`), which produced `evaluated/evaluated/evaluated/...` | Output folders get a `.evaluation_output` marker. Folders with the marker or named `evaluated*` are skipped. |
| E2 **[M]** | `test_validation/` contains two runs per model (28 and 29 May). The aggregates averaged both, so each submission counted twice | Duplicates of (model, prompt type, submission) are dropped, keeping the most recent run (with a warning). `--keep-duplicates` turns this off. New columns `source_run`, `source_file`. |
| E3 **[M]** | Invalid-JSON rows are excluded from the metrics (Eberhard's rule), so a model that often fails the format looks no worse | New `format_adherence_by_model_prompttype.csv`: valid JSON rate, valid structure rate, truncation rates, mean number of recommendations. Report it next to the metrics. |

### F. Docker / Compose

| ID | Change | Rationale |
|---|---|---|
| F1 | Compose files: `test_all_models` (updated), new `full_run` and `openai_run`. Run settings as env vars (`RUN_NAME`, `MODEL_SETUP`, `PROMPT_TYPES`, `LIMIT`, `INPUT_MODE`) | One place to see what a run does. The GPT run needs no cluster. |
| F2 | Each compose file uses its own service names (`mongodb_test`/`recommendation_test`, `…_full`, `…_openai`). Fixed `container_name`s removed | Two files both used `container_name: mongodb` and could not run at the same time. A top-level `name:` was tried first, but the server's Docker Compose v2.2.3 rejects it; all files were validated with `docker compose config` on funkenzutzler. |
| F3 | MongoDB: `mongoimport --drop`; health check only passes after the import has finished; image pinned to `mongo:8.0` (was `latest`) | No duplicates or partial data after restarts. The experiment no longer starts while the import is still running. Pinned version for reproducibility. |
| F4 | Dataset path configurable (`DATASET_FILE`) | The local and server folder layouts differ. |
| F5 | Dockerfile: `PYTHONUNBUFFERED=1`, `TZ=Europe/Vienna`; unused packages removed (matplotlib, seaborn, tabulate, jsonschema) | Live logs in `docker logs`; one time zone in all logs (run.log used UTC before). |

---

## 3. Comparability with the test runs of 28/29 May 2026

The existing test results in `setup/results/test_validation/2026_05_*` were produced **before** these
changes. If they are reported, note that they differ from new runs in:
- context window 4096 (now 8192), no seed (now 42);
- the 10 submissions were chosen by MongoDB's natural order (now `_id` order), so they may be different submissions;
- the gpt-oss models are missing (not available);
- title only (now title + body text);
- the old CSV format (no `valid_structure`, `finish_reason`, `duration_seconds`, `timestamp`).
- output-token limit 1000 (now 4096), and Ollama 0.7.1 (now 0.35.0).

Prompts, models, quantisation and the evaluation metrics are unchanged.

## 4. Verification

All checks were run on 2026-10-01 in the experiment Docker image (Python 3.13, same package versions):

| Test | Result |
|---|---|
| `tests/test_units.py`: 31 offline tests (parsing, few-shot determinism, resume after a simulated tunnel failure, retry logic, OpenAI routing, progress states, squeue parsing, job monitoring, evaluator de-duplication and output skipping) | all pass |
| End-to-end: real MongoDB (compose file, full dataset → 1,718 documents imported, health check), real Ollama 0.35 with `llama3.2:1b` on CPU, `run_models.py`, 5 submissions × zero-/few-shot | 5 rows per prompt type written, 100 % valid JSON. A non-existent model reported as `FAIL (exit 2)`. Container exit code 2 |
| Resume: same run started again | model skipped, no LLM calls |
| Evaluator: run twice on the same folder | no nested outputs. Format adherence file written |
| Scheduler against a simulated cluster (fake `ssh`/`sbatch`/`squeue`/`scancel`) | upload as `ollama_serve_<port>.sh`, PENDING → RUNNING, tunnel, readiness wait, script exit code 7 passed through, `scancel` at the end |
| `docker stop` during a running script | SIGTERM handled, `scancel` executed, exit 130 |

Not verified locally: the real connection to nvcluster (requires the institute network). Run the test
compose file on funkenzutzler first (docker README §2).

## 5. Open items

- **Update Ollama on nvcluster** (`~/bin/ollama`, currently 0.7.1) so that `gpt-oss:20b/120b` can be pulled (C4).
- **Deploy** the changed `experiment/` folder to funkenzutzler and run the test compose file before the full run.
- `NUM_THREAD = 2` (CPU threads for layers not on the GPU) was inherited and left unchanged. If the
  123B model is partly offloaded to CPU, more threads could speed it up. It does not affect outputs.
- The private key `TUServer`/`TUServer.ppk` was copied to `%USERPROFILE%/.ssh/` (access restricted to the user) on 2026-10-01. The copies in `Downloads/Master/Master/` can be deleted.
- The repository's last commit dates from the data-cleaning phase; `experiment/`, `docs/`, `analysis/`, `archive/` are untracked. Commit them to version the experiment code used for the thesis.

## 6. Server check (2026-10-01)

Read-only inspection of funkenzutzler and nvcluster:

| Finding | Consequence |
|---|---|
| `~/experiment` on funkenzutzler is identical to the local state *before* the clean-up (same checksums for dataset, prompts, scripts) | Deploying the new `experiment/` folder is all that is needed. The 28/29 May test results there are the same as the local copy |
| Docker Compose v2.2.3 (rootless Docker) | `name:` removed from the compose files (F2) |
| funkenzutzler → nvcluster SSH (`~/.ssh/id_rsa`) works without a password; no jobs running | Prerequisite for the scheduler is fulfilled |
| nvcluster `~/bin/ollama` = 0.7.1 | Explains the gpt-oss failure (C4) |
| Partition `hcc`: 2 nodes × 8 NVIDIA Quadro RTX 8000 (48 GB), time limit infinite | Hardware for the methods chapter. 2 GPUs = 96 GB VRAM for the 123B model. No SLURM time-outs expected |
| `setup/docker/.env` contains `OPENAI_API_KEY` (the line starts with a space) | Compose reads it automatically. Remove the leading space if the key is not picked up |

## 7. Deployment (2026-10-01)

| Step | What was done |
|---|---|
| Ollama update on nvcluster | `~/bin/ollama` + `~/lib/ollama` updated from **0.7.1 to 0.35.0** (official release `ollama-linux-amd64.tar.zst`, pinned version). Old version kept in `~/ollama-0.7.1-backup/`. Downloaded models in `~/.ollama` (237 GB) unchanged. GPU check from earlier logs: Quadro RTX 8000, compute capability 7.5, driver supports CUDA 13.0. Compute nodes: `nvcluster-node2`, `nvcluster-node4` (628 GB RAM each) |
| Code on funkenzutzler | Old `~/experiment` moved to `~/experiment_backup_2026-10-01`. New `experiment/` code uploaded (without local results). `dataset/`, `setup/results/`, `log/` and `setup/docker/.env` copied over from the backup. Checksums of the deployed files verified against the local copy |
| Smoke test | New `docker-compose.smoke_test.yml` (llama3.2:3b + gpt-oss:20b, 3 submissions, zero- + few-shot, 1 GPU, port 28023, results in `setup/results/smoke_test/`) as a first end-to-end check on the real infrastructure before the 12-model test run |

## 8. Smoke test findings and fixes (2026-10-01)

**Smoke test 1** (`setup/results/smoke_test_1000tokens/`, kept as evidence): infrastructure fully working
(SLURM job 132937 on `nvcluster-node2`, Quadro RTX 8000, driver 580.105.08, Ollama 0.35.0; tunnel, readiness
check, job cancellation). llama3.2:3b: 6/6 valid answers. **gpt-oss:20b: 0/6 usable answers** – every
response stopped at exactly 1000 tokens (`finish_reason = length`), 4 empty, 2 with JSON cut off mid-list.
Cause: reasoning models generate internal "thinking" tokens first; they count towards `num_predict` but are
returned separately, not in the answer. The run summary nevertheless reported "OK".

| ID | Change |
|---|---|
| D17 **[M]** | `MAX_OUTPUT_TOKENS` (Ollama `num_predict`) 1000 → **4096** for all models (decision in §1). Also relevant for deepseek-r1 and qwen3, which are reasoning models too and may now behave the same way on Ollama 0.35 (they stayed under 1000 tokens with Ollama 0.7.1 in May). |
| B14 | `run_models.py`: after each model, the share of valid-JSON and truncated answers per prompt type is computed from the result CSVs and written to `run.log`; a model with < 50 % valid JSON is marked **WARN** in the summary (exit code unchanged). |

## 9. Model list extended (2026-10-02)

| ID | Change | Rationale |
|---|---|---|
| C6 **[M]** | 8 newer Ollama models added (all q4_K_M, tags verified in the Ollama library): `qwen3.5` 4b / 9b / 27b / 122b-a10b, `gemma4` e4b-it / 31b-it, `olmo-3:7b-instruct`, `olmo-3.1:32b-instruct`. All 12 original models kept. New field `generation` (1 = original, 2 = added) in `models.py` | qwen3.5 spans all size classes (effect of model size within one family); gemma4 succeeds gemma3 (generation comparison); OLMo is fully open incl. training data (discussion of training-data overlap with Reddit). Size classes follow the parameter rule (< 4B tiny, 4–10B small, …), so qwen3.5:4b and gemma4:e4b are "small" |
| C7 | New setup `tiny_and_small_models` (8 models that fit on one 48 GB GPU) | First batch of the full run with a small footprint on the shared cluster; GPU usage is checked before larger batches are scheduled |

## 10. Repository clean-up (2026-10-02)

| Change | Rationale |
|---|---|
| `legacy/` renamed to `dataset_creation/` | It contains the pipeline that produced the dataset and ground truth (filtering steps, GPT labelling batches) - methodologically relevant, not obsolete |
| Hard-coded OpenAI API key removed from 11 `dataset_creation/gpt_batch_scripts/**/handle_batches.py`; read from `OPENAI_API_KEY` instead | Secret in source code. Two *older* keys had already been pushed to GitHub in March 2025 (commit `aeaa16d`) and should be revoked |
| `.gitignore` rewritten: excludes `data/` (raw data, ~5.4 GB), experiment results/logs, `.env`, `.venv`, `.idea` and other editor settings, `archive/results_stale/`. `.gitattributes` forces LF for `.sh`/`.py`/`.yml`/`Dockerfile` | The old rules only matched folders at the repo root, so after the restructuring the raw data would have been committed. LF: scripts are copied to Linux servers from a Windows checkout |
| `experiment/python-llm-toolkit/` vendored: its nested `.git` moved outside the repository (local backup), upstream noted at the top of its `README.md` | A nested repository is stored by git only as a pointer, so the modified scheduler code would not have been versioned |
| Early scripts and the first manual GPT labelling experiment moved into `archive/early_scripts/`; broken `.venv` and empty `results/` removed | Files existed only in the old local working copy |
| Outside the repository: `literature/`, `notes/`, `personal/` created; nvcluster password removed from `notes/commands.txt` | Order; plain-text password |
| Documentation renamed to lowercase (`docs/experiment-pipeline.md`, `docs/changelog.md`, `experiment/setup/docker/README.md`); project structure merged into the root `README.md`; toolkit origin noted in its `README.md`. Planning notes (roadmap, status summaries, writing phrases, thesis outline) and the superseded guides moved out of the repository to `notes/planning-docs/` | Consistent naming; the repository contains only project documentation |
| Emojis removed from analysis scripts, `analysis/README.md` and `tests/dry_run_test.py` (status prints now `[OK]` / `[FAIL]` / `[WARN]`). `analysis/scripts/comprehensive_classification_analysis.py` → `classification_stats.py` (output `classification_stats.json`), `precise_classification_analysis.py` → `classification_missing_types.py`. Prose reports (`iteration_analysis.md`, `mongodb_iteration_correlation_analysis.md`, `prompt_engineering_report.md`) moved to `notes/reports/` (outside the repository). Prompt zips unpacked to `archive/prompt_versions/v1`, `v2` | Plain, descriptive names and output; old prompt versions readable as text |
