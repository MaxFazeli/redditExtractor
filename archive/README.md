# Archive

Files that are no longer part of the experiment pipeline. They were moved here on 2026-10-01
during the pipeline clean-up (see `docs/changelog.md`). Nothing was deleted.

## `bug_insertion/`
Left-overs from a colleague's project ("bug insertion" into Python code with LLMs). The
orchestration code (`python-llm-toolkit/sbatch_scheduler`) and the framework structure of this
experiment were derived from it, but these files were never used for the recommendation experiment.

| File | Original location | Why archived |
|---|---|---|
| `setup_scripts/bug_insertion.sh`, `tiny_models*.sh`, `small_models.sh`, `medium_models*.sh`, `large_models*.sh` | `experiment/setup/scripts/` | Run `02_bug_insertion.py`, which does not exist in this project |
| `setup_scripts/ollama_simple.sh` | `experiment/setup/scripts/` | Bug-insertion entry point with the colleague's cluster user |
| `docker/docker-compose.yml` | `experiment/setup/docker/` | Mounted `cs-master/...` bug-insertion sources; used the invalid port 71195 |
| `framework/models.py` | `experiment/framework/` | Original model catalogue (coding models: starcoder, codellama, ...). Replaced by the thesis model list |
| `framework/config.py` | `experiment/framework/` | Original version with bug-insertion constants (`BUG_INSERTION_DATA_DIRECTORY`, code-diff keys, ...) |
| `framework/json_utils.py` | `experiment/framework/` | Original version; referenced `json_response_schema.json` of the bug-insertion task |
| `framework/progress_utils.py` | `experiment/framework/` | Original version (non-atomic writes, no failed/not-available states) |
| `framework/utils.py` | `experiment/framework/` | CSV/duration helpers not used by the recommendation experiment |

## `superseded/`
Earlier versions of the recommendation experiment that were replaced.

| File | Original location | Replaced by |
|---|---|---|
| `scripts/recommendation_experiment_lite.py` | `experiment/` | `recommendation_experiment.py --limit N --model X` (the lite script also called a non-existent `openai_client.chat()`) |
| `scripts/movie_tv_zero_shot_test.txt` | `experiment/prompts/zero_shot_prompts/` | Only used by the lite script |
| `scripts/run_test_all_models.py` | `experiment/setup/scripts/` | `setup/scripts/run_models.py` (configurable via env vars, resumable, correct exit status) |
| `scripts/run_experiment.sh` | `experiment/setup/scripts/` | `setup/scripts/run_models.py` |
| `docker/docker-compose.recommendation.yml` | `experiment/setup/docker/` | `docker-compose.full_run.yml` / `docker-compose.openai_run.yml` (old file used invalid port 71195 and the lite script) |

## `results_stale/`
| Folder | Original location | Why archived |
|---|---|---|
| `test_validation_new/` | `experiment/setup/results/` | Byte-identical copy of the 2026-05-29 runs that are also in `test_validation/` |
| `test_validation_evaluated/`, `test_validation_evaluated_new/` | `experiment/setup/results/test_validation/` | Evaluations made with the old evaluator: they mix the 28 May and 29 May runs (each model counted twice) and contain recursively nested `evaluated/evaluated/...` copies. Re-create with the current `recommendation_evaluator.py`. |

The raw test-run results themselves (`experiment/setup/results/test_validation/2026_05_*`) were left in place.

## `bug_insertion/max_setup/`
Former top-level folder `max/`: the colleague's complete bug-insertion setup (Docker/compose files,
`prompts.json` with bug-insertion prompts, model scripts). Not used by the recommendation experiment.

## `early_scripts/` (added 2026-10-02)
From the old local working copy (`Downloads/Master/Master/Master/`), files that existed nowhere in the repository:
- `submissions/sub_1` … `sub_5`, `domains_distribution.txt` - first versions of the submission filtering steps (later renumbered, see `dataset_creation/scripts/`)
- `requestGPT.txt`, `outputChatGPT4omini.txt` - first manual GPT-4o-mini labelling experiment (prompt + output)

## Not in git
`results_stale/` is git-ignored (duplicated / outdated evaluation outputs); it stays only on the local disk.

Outdated documentation and planning notes (old orchestration guide, setup guides, roadmaps, status
summaries, writing notes) were moved out of the repository to the local `notes/planning-docs/` folder on 2026-10-02.

## `prompt_versions/` (added 2026-10-02)
Earlier versions of the zero-/few-shot prompts (`v1`, `v2`), previously stored as zip files in
`experiment/prompts/`. The current prompts are in `experiment/prompts/zero_shot_prompts/` and `few_shot_prompts/`.
