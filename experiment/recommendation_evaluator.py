#!/usr/bin/env python3
# encoding: utf-8

"""
Recommendation Evaluator

Reads result CSVs produced by recommendation_experiment.py and computes evaluation metrics.

Matching strategy (in priority order):
  1. ID match  — IMDb ID for Film/TV, ISBN for Books (exact, case-insensitive)
  2. Exact title match (case-insensitive, stripped)
  3. Fuzzy title match (SequenceMatcher ratio >= 0.9)  [fuzzy mode only]

Metrics computed:
  - exact_match: precision, recall, F1, nDCG  (ID match OR exact title)
  - fuzzy_match: precision, recall, F1, nDCG  (ID match OR exact title OR fuzzy title)

Rules per Eberhard:
  - Only use the first 10 unique recommendations from each LLM response
  - Skip rows where valid_json is False or nr_true_recs < 5
  Because invalid responses are excluded from the metrics, format adherence (valid JSON rate,
  valid structure rate, truncation rate) is reported separately in format_adherence_by_model_prompttype.csv.

Duplicates:
  If the same (model, prompt_type, submission_id) occurs in several runs below --results-dir
  (e.g. a test run repeated twice), only the most recent row is kept, unless --keep-duplicates is given.

Usage:
    python recommendation_evaluator.py --results-dir ./results
    python recommendation_evaluator.py --results-dir ./results --output-dir ./evaluated
"""

import os
import sys
import json
import argparse
from math import log
from difflib import SequenceMatcher

import pandas as pd

NR_RECOMMENDATIONS = 10
FUZZY_THRESHOLD = 0.9

# Written into every evaluation output directory so that it is never scanned as input again
# (previously, re-running the evaluator produced evaluated/evaluated/evaluated/...).
OUTPUT_MARKER = '.evaluation_output'
DUPLICATE_KEY = ['model', 'prompt_type', 'submission_id']

# Which field to use for ID-based matching per domain
DOMAIN_ID_FIELD = {
    'Film': 'IMDb',
    'TV':   'IMDb',
    'Books': 'ISBN',
    'Music': None,
    'Games': None,
}


def normalize_title(title) -> str:
    if not title:
        return ''
    return str(title).lower().strip()


def normalize_id(val) -> str:
    if not val:
        return ''
    return str(val).strip().lower().replace('-', '').replace(' ', '')


def extract_item(rec: dict, domain: str) -> dict:
    title = normalize_title(rec.get('title', ''))
    id_field = DOMAIN_ID_FIELD.get(domain)
    item_id = ''
    if id_field:
        raw_id = rec.get(id_field) or rec.get(id_field.lower()) or ''
        item_id = normalize_id(raw_id)
    return {'title': title, 'id': item_id}


def is_match(pred: dict, true: dict, fuzzy: bool) -> bool:
    if pred['id'] and true['id'] and pred['id'] == true['id']:
        return True
    if pred['title'] and true['title'] and pred['title'] == true['title']:
        return True
    if fuzzy and pred['title'] and true['title']:
        if SequenceMatcher(None, pred['title'], true['title']).ratio() >= FUZZY_THRESHOLD:
            return True
    return False


def compute_metrics(pred_items: list, true_items: list, fuzzy: bool) -> dict:
    tp = 0
    DCG = 0.0
    IDCG = 0.0

    for i, pred in enumerate(pred_items, 1):
        if any(is_match(pred, t, fuzzy) for t in true_items):
            tp += 1
            DCG += 1.0 / log(i + 1, 2)

    fp = len(pred_items) - tp
    fn = len(true_items) - tp

    for i in range(1, min(len(true_items), len(pred_items)) + 1):
        IDCG += 1.0 / log(i + 1, 2)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    ndcg      = DCG / IDCG if IDCG > 0 else 0.0

    return {'precision': precision, 'recall': recall, 'f1': f1, 'nDCG': ndcg}


def safe_parse_json(val) -> list:
    if val is None or (isinstance(val, float)):
        return []
    try:
        result = json.loads(str(val))
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            return result.get('recommendations', [result])
        return []
    except Exception:
        return []


def dedup_preserve_order(items: list) -> list:
    seen = set()
    out = []
    for item in items:
        key = item['title']
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def evaluate_row(row: pd.Series) -> dict:
    domain = str(row.get('domain', ''))

    llm_raw  = safe_parse_json(row.get('llm_recommendations'))
    true_raw = safe_parse_json(row.get('ground_truth'))

    pred_all  = [extract_item(r, domain) for r in llm_raw  if isinstance(r, dict)]
    true_all  = [extract_item(r, domain) for r in true_raw if isinstance(r, dict)]

    pred_deduped = dedup_preserve_order(pred_all)
    pred_used    = pred_deduped[:NR_RECOMMENDATIONS]

    id_match_count = sum(
        1 for p in pred_used
        if p['id'] and any(t['id'] and p['id'] == t['id'] for t in true_all)
    )

    exact = compute_metrics(pred_used, true_all, fuzzy=False)
    fuzzy = compute_metrics(pred_used, true_all, fuzzy=True)

    return {
        'nr_recs_used':              len(pred_used),
        'nr_true_recs_eval':         len(true_all),
        'id_match_count':            id_match_count,
        'exact_match_precision':     exact['precision'],
        'exact_match_recall':        exact['recall'],
        'exact_match_f1':            exact['f1'],
        'exact_match_nDCG':          exact['nDCG'],
        'fuzzy_match_precision':     fuzzy['precision'],
        'fuzzy_match_recall':        fuzzy['recall'],
        'fuzzy_match_f1':            fuzzy['f1'],
        'fuzzy_match_nDCG':          fuzzy['nDCG'],
    }


EVALUATION_COLUMNS = [
    'nr_recs_used', 'nr_true_recs_eval', 'id_match_count',
    'exact_match_precision', 'exact_match_recall', 'exact_match_f1', 'exact_match_nDCG',
    'fuzzy_match_precision', 'fuzzy_match_recall', 'fuzzy_match_f1', 'fuzzy_match_nDCG',
]


def evaluate_csv(input_path: str, output_path: str, source_run: str = '') -> pd.DataFrame:
    df = pd.read_csv(input_path)

    metric_rows = [evaluate_row(row) for _, row in df.iterrows()]
    metrics_df  = pd.DataFrame(metric_rows, columns=EVALUATION_COLUMNS)
    result_df   = pd.concat([df.reset_index(drop=True), metrics_df], axis=1)
    result_df['source_run'] = source_run
    result_df['source_file'] = input_path

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    result_df.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}  ({len(result_df)} rows)")
    return result_df


def drop_duplicate_runs(combined: pd.DataFrame) -> pd.DataFrame:
    """Keep only the most recent row per (model, prompt_type, submission_id)."""
    if not all(k in combined.columns for k in DUPLICATE_KEY):
        return combined
    order_cols = [c for c in ('timestamp', 'source_run') if c in combined.columns]
    # rows without timestamp (older CSV format) count as oldest
    ordered = combined.sort_values(order_cols, kind='stable', na_position='first') if order_cols else combined
    deduped = ordered.drop_duplicates(subset=DUPLICATE_KEY, keep='last')
    dropped = len(combined) - len(deduped)
    if dropped:
        runs = combined.groupby(['model', 'prompt_type'])['source_run'].nunique()
        print(f"\n  WARNING: {dropped} duplicate rows from repeated runs dropped (kept the most recent run).")
        for (model, prompt_type), n in runs[runs > 1].items():
            print(f"    {model} / {prompt_type}: {n} runs")
    return deduped.sort_index()


def format_adherence(combined: pd.DataFrame) -> pd.DataFrame:
    """Share of responses that are valid JSON / follow the requested structure / were cut off, per model and prompt type."""
    df = combined.copy()
    for col in ('valid_json', 'valid_structure', 'response_truncated', 'prompt_truncated'):
        if col in df.columns:
            df[col] = df[col].map(lambda v: str(v).strip().lower() == 'true')
    agg = {'submission_id': 'count'}
    for col in ('valid_json', 'valid_structure', 'response_truncated', 'prompt_truncated', 'nr_recs', 'nr_recs_unique'):
        if col in df.columns:
            agg[col] = 'mean'
    out = df.groupby(['model', 'prompt_type']).agg(agg).rename(columns={
        'submission_id': 'rows',
        'valid_json': 'valid_json_rate',
        'valid_structure': 'valid_structure_rate',
        'response_truncated': 'response_truncated_rate',
        'prompt_truncated': 'prompt_truncated_rate',
        'nr_recs': 'mean_nr_recs',
        'nr_recs_unique': 'mean_nr_recs_unique',
    })
    return out.round(4)


def aggregate_results(all_dfs: list, output_dir: str, keep_duplicates: bool = False):
    if not all_dfs:
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    if not keep_duplicates:
        combined = drop_duplicate_runs(combined)

    if all(k in combined.columns for k in ('model', 'prompt_type', 'submission_id')):
        format_adherence(combined).to_csv(os.path.join(output_dir, 'format_adherence_by_model_prompttype.csv'))

    # Apply quality filter: valid JSON and ≥5 true recommendations
    valid = combined[
        (combined.get('valid_json', pd.Series([True] * len(combined))) == True) &
        (combined['nr_true_recs_eval'] >= 5)
    ].copy()

    print(f"\n  Total rows: {len(combined)}  |  After quality filter: {len(valid)}")

    metric_cols = [
        'exact_match_precision', 'exact_match_recall', 'exact_match_f1', 'exact_match_nDCG',
        'fuzzy_match_precision', 'fuzzy_match_recall', 'fuzzy_match_f1', 'fuzzy_match_nDCG',
    ]

    group_keys = [
        (['model', 'prompt_type'],          'aggregate_by_model_prompttype.csv'),
        (['model', 'prompt_type', 'domain'], 'aggregate_by_model_prompttype_domain.csv'),
        (['domain'],                         'aggregate_by_domain.csv'),
        (['prompt_type'],                    'aggregate_by_prompttype.csv'),
    ]

    for keys, filename in group_keys:
        if not all(k in valid.columns for k in keys):
            continue
        agg = valid.groupby(keys)[metric_cols].agg(['mean', 'std', 'count']).round(4)
        agg.columns = ['_'.join(c) for c in agg.columns]
        agg.to_csv(os.path.join(output_dir, filename))

    # Full combined evaluated CSV for further analysis
    valid.to_csv(os.path.join(output_dir, 'all_evaluated.csv'), index=False)

    print(f"\nAggregate files written to: {output_dir}")
    print("\n--- F1 by Model × Prompt Type (fuzzy) ---")
    if 'model' in valid.columns and 'prompt_type' in valid.columns:
        summary = valid.groupby(['model', 'prompt_type'])[['fuzzy_match_f1', 'fuzzy_match_nDCG']].mean().round(4)
        print(summary.to_string())


def find_result_csvs(results_dir: str, output_dir: str = None) -> list:
    output_dir_abs = os.path.abspath(output_dir) if output_dir else None
    found = []
    for root, dirs, files in os.walk(results_dir):
        # Skip the current output directory and any earlier evaluation output (marker file or
        # "evaluated*" folder name) to avoid re-evaluating already-evaluated files
        root_abs = os.path.abspath(root)
        if (output_dir_abs and (root_abs == output_dir_abs or root_abs.startswith(output_dir_abs + os.sep))) \
                or OUTPUT_MARKER in files:
            dirs.clear()
            continue
        dirs[:] = sorted(d for d in dirs if not d.startswith('evaluated'))
        for f in sorted(files):
            if f.endswith('_results.csv') or f == 'results.csv':
                found.append(os.path.join(root, f))
    return found


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate LLM recommendation results against ground truth.'
    )
    parser.add_argument('--results-dir', required=True,
                        help='Directory containing result CSVs (searched recursively)')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory (default: <results-dir>/evaluated)')
    parser.add_argument('--keep-duplicates', action='store_true',
                        help='Keep rows of repeated runs instead of only the most recent one')
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.results_dir, 'evaluated')
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, OUTPUT_MARKER), 'w') as f:
        f.write('Output of recommendation_evaluator.py - not scanned as input.\n')

    csv_files = find_result_csvs(args.results_dir, output_dir)
    if not csv_files:
        print(f"No result CSVs found in {args.results_dir}")
        sys.exit(1)

    print(f"Found {len(csv_files)} result CSV(s)")

    all_dfs = []
    for csv_path in csv_files:
        rel = os.path.relpath(csv_path, args.results_dir)
        out = os.path.join(output_dir, rel)
        print(f"\nEvaluating: {rel}")
        source_run = rel.split(os.sep)[0] if os.sep in rel else ''
        df = evaluate_csv(csv_path, out, source_run=source_run)
        all_dfs.append(df)

    print("\nAggregating results...")
    aggregate_results(all_dfs, output_dir, keep_duplicates=args.keep_duplicates)


if __name__ == '__main__':
    main()
