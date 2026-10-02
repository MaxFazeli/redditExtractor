#!/usr/bin/env python3
# encoding: utf-8

"""
Test Orchestrator — validates the full pipeline for every Ollama model.
Runs 10 submissions × zero_shot per model, sequentially.

Called by sbatch_scheduler.py as --script (no args allowed — all config is hardcoded).
Ollama is already running on nvcluster with port forwarding when this starts.
"""

import subprocess
import sys
import os
from datetime import datetime

SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_DIR  = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
EXPERIMENT_SCRIPT = os.path.join(EXPERIMENT_DIR, 'recommendation_experiment.py')
OUTPUT_DIR      = '/app/results/test_validation'
LOG_FILE        = os.path.join(OUTPUT_DIR, 'test_run.log')

# All Ollama models grouped by size (GPT models handled separately via API)
MODELS = [
    # Tiny (< 4B)
    'llama3.2:3b-instruct-q4_K_M',

    # Small (4-10B)
    'gemma3:4b-it-q4_K_M',
    'llama3.1:8b-instruct-q4_K_M',
    'deepseek-r1:8b-0528-qwen3-q4_K_M',

    # Medium (10-50B)
    'gpt-oss:20b',
    'mistral-small3.2:24b-instruct-2506-q4_K_M',
    'gemma3:27b-it-q4_K_M',
    'qwen3:32b-q4_K_M',
    'deepseek-r1:32b-qwen-distill-q4_K_M',

    # Large (≥ 50B)
    'llama3.3:70b-instruct-q4_K_M',
    'gpt-oss:120b',
    'mistral-large:123b-instruct-2411-q4_K_M',
]

LIMIT = 10


def log(msg: str):
    ts   = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def run_model(model: str) -> bool:
    log(f"START  {model}")
    t0 = datetime.now()

    result = subprocess.run(
        [
            sys.executable, EXPERIMENT_SCRIPT,
            '--model',        model,
            '--limit',        str(LIMIT),
            '--prompt-types', 'zero_shot',
            '--output-dir',   OUTPUT_DIR,
        ],
        cwd=EXPERIMENT_DIR,
    )

    elapsed = (datetime.now() - t0).seconds
    status  = 'OK' if result.returncode == 0 else f'FAIL (exit {result.returncode})'
    log(f"DONE   {model}  →  {status}  [{elapsed}s]")
    return result.returncode == 0


def main():
    log(f"=== TEST RUN: {len(MODELS)} models, limit={LIMIT}, zero_shot ===")
    log(f"Output dir: {OUTPUT_DIR}")

    results = {}
    for model in MODELS:
        results[model] = run_model(model)

    log("=== SUMMARY ===")
    for model, ok in results.items():
        mark = 'OK  ' if ok else 'FAIL'
        log(f"  [{mark}]  {model}")

    passed = sum(results.values())
    log(f"=== {passed}/{len(MODELS)} models completed successfully ===")


if __name__ == '__main__':
    main()
