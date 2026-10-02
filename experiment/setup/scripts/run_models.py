#!/usr/bin/env python3
# encoding: utf-8

"""
Run orchestrator — runs recommendation_experiment.py once per model, sequentially.

Started by sbatch_scheduler.py (--script) after Ollama is running on nvcluster and the SSH
tunnel is up. The scheduler cannot pass arguments, so the run is configured via environment
variables (set in the docker-compose file):

    RUN_NAME       name of the run; results go to $RESULTS_DIR/$RUN_NAME/   (default: run)
    MODEL_SETUP    model setup key from framework/models.py                (default: ollama_models)
    MODELS         comma-separated model keys; overrides MODEL_SETUP
    PROMPT_TYPES   space- or comma-separated prompt types                  (default: zero_shot few_shot)
    LIMIT          only use the first N submissions                        (default: all)
    INPUT_MODE     title | title_selftext                                  (default: title_selftext)
    RESULTS_DIR    results root inside the container                       (default: /app/results)

Resuming: every model gets a fixed progress file $RESULTS_DIR/$RUN_NAME/progress_<model>.json.
Starting the same RUN_NAME again skips finished models and finished submissions.

Exit code: 0 if every model finished, otherwise the first non-zero exit code of the experiment
script (2 = model not available, 3 = infrastructure error, 4 = configuration error).
"""

import csv
import glob
import os
import subprocess
import sys
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
EXPERIMENT_SCRIPT = os.path.join(EXPERIMENT_DIR, 'recommendation_experiment.py')
sys.path.insert(0, os.path.join(EXPERIMENT_DIR, 'framework'))

import models as models_config  # noqa: E402

EXIT_INFRASTRUCTURE_ERROR = 3

# A model whose answers are mostly unusable is marked WARN in the summary (the run itself
# succeeded, but the results need a look - e.g. reasoning models running out of tokens).
MIN_VALID_JSON_RATE = 0.5

csv.field_size_limit(10 ** 8)

RUN_NAME = os.environ.get('RUN_NAME', 'run')
RESULTS_DIR = os.environ.get('RESULTS_DIR', '/app/results')
OUTPUT_DIR = os.path.join(RESULTS_DIR, RUN_NAME)
LOG_FILE = os.path.join(OUTPUT_DIR, 'run.log')


def _split(value):
    return [v for v in value.replace(',', ' ').split() if v]


def get_models():
    if os.environ.get('MODELS'):
        return _split(os.environ['MODELS'])
    return models_config.get_models_for_setup(os.environ.get('MODEL_SETUP', 'ollama_models'))


def log(msg: str):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def answer_quality(model):
    """Share of valid-JSON and truncated answers of a model in this run, per prompt type.

    Returns {prompt_type: (rows, valid_json_rate, truncated_rate)}.
    """
    quality = {}
    for path in sorted(glob.glob(os.path.join(OUTPUT_DIR, models_config.model_slug(model), '*_results.csv'))):
        prompt_type = os.path.basename(path)[:-len('_results.csv')]
        with open(path, newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        valid = sum(r.get('valid_json') == 'True' for r in rows) / len(rows)
        truncated = sum(r.get('response_truncated') == 'True' for r in rows) / len(rows)
        quality[prompt_type] = (len(rows), valid, truncated)
    return quality


def quality_status(quality):
    """'OK', or 'WARN (...)' if any prompt type has mostly invalid answers."""
    details = ', '.join(f"{pt}: {v:.0%} valid JSON, {t:.0%} truncated (n={n})" for pt, (n, v, t) in quality.items())
    if any(v < MIN_VALID_JSON_RATE for _, v, _ in quality.values()):
        return f"WARN - mostly unusable answers ({details})"
    return f"OK ({details})" if details else 'OK'


def run_model(model, prompt_types, limit, input_mode) -> int:
    progress_file = os.path.join(OUTPUT_DIR, f"progress_{models_config.model_slug(model)}.json")
    cmd = [
        sys.executable, EXPERIMENT_SCRIPT,
        '--model', model,
        '--prompt-types', *prompt_types,
        '--progress-file', progress_file,
        '--input-mode', input_mode,
    ]
    if limit:
        cmd += ['--limit', str(limit)]

    log(f"START  {model}")
    t0 = datetime.now()
    result = subprocess.run(cmd, cwd=EXPERIMENT_DIR)
    elapsed = int((datetime.now() - t0).total_seconds())
    if result.returncode == 0:
        status = quality_status(answer_quality(model))
    else:
        status = {2: 'NOT AVAILABLE', 3: 'INFRASTRUCTURE ERROR', 4: 'CONFIG ERROR'}.get(
            result.returncode, f'FAIL (exit {result.returncode})')
    log(f"DONE   {model}  ->  {status}  [{elapsed}s]")
    return result.returncode


def main():
    models = get_models()
    prompt_types = _split(os.environ.get('PROMPT_TYPES', 'zero_shot few_shot'))
    limit = os.environ.get('LIMIT') or None
    input_mode = os.environ.get('INPUT_MODE', 'title_selftext')

    log(f"=== RUN '{RUN_NAME}': {len(models)} models, prompt types={prompt_types}, "
        f"limit={limit or 'all'}, input_mode={input_mode} ===")
    log(f"Output dir: {OUTPUT_DIR}")

    results = {}
    for model in models:
        results[model] = run_model(model, prompt_types, limit, input_mode)
        if results[model] == EXIT_INFRASTRUCTURE_ERROR:
            log("Infrastructure error (Ollama/tunnel/API unreachable) - stopping the run. "
                "Start the same RUN_NAME again to resume.")
            break

    log("=== SUMMARY ===")
    warnings = 0
    for model in models:
        code = results.get(model)
        if code == 0:
            status = quality_status(answer_quality(model))
            mark = 'WARN' if status.startswith('WARN') else 'OK  '
            warnings += mark == 'WARN'
            log(f"  [{mark}]  {model}  {status}")
        else:
            mark = 'SKIP' if code is None else 'FAIL'
            log(f"  [{mark}]  {model}" + (f"  (exit {code})" if code is not None else ''))
    passed = sum(1 for c in results.values() if c == 0)
    log(f"=== {passed}/{len(models)} models completed successfully"
        + (f", {warnings} with mostly unusable answers (WARN)" if warnings else '') + " ===")

    failures = [c for c in results.values() if c != 0]
    return failures[0] if failures else 0


if __name__ == '__main__':
    sys.exit(main())
