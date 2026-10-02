#!/usr/bin/env python
# encoding: utf-8

"""
Recommendation Experiment Script
Generates recommendations for r/ifyoulikeblank requests using various LLMs and stores them
together with the human ground truth (Reddit comments) for later evaluation.

Originally derived from a colleague's bug-insertion framework (02_bug_insertion.py); the
model loop / progress-file concept is kept, everything else is specific to this experiment.

Exit codes (used by setup/scripts/run_models.py):
    0  all requested models finished
    2  at least one model was not available (could not be pulled / unknown API model)
    3  infrastructure error (Ollama/OpenAI unreachable after retries) - run can be resumed
    4  configuration error (bad arguments, MongoDB unreachable, empty dataset, ...)
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(EXPERIMENT_DIR, 'framework'))
sys.path.insert(0, EXPERIMENT_DIR)

from pymongo import MongoClient
from pymongo.errors import PyMongoError
from tqdm import tqdm
from ollama._types import ResponseError

import config
import json_utils
import progress_utils
import models as models_config
from logger import debug_msg, logging
from ollama_client import OllamaClient, InfrastructureError, TRANSPORT_ERRORS
from openai_client import OpenAiClient, INFRASTRUCTURE_ERRORS as OPENAI_INFRASTRUCTURE_ERRORS
from recommendation_utils import ResultWriter, create_directory_if_not_exists

EXIT_OK = 0
EXIT_MODEL_NOT_AVAILABLE = 2
EXIT_INFRASTRUCTURE_ERROR = 3
EXIT_CONFIG_ERROR = 4

# MongoDB connection — env vars so the Docker service name ("mongodb") works
MONGO_HOST = os.environ.get('MONGODB_HOST', 'localhost')
MONGO_PORT = int(os.environ.get('MONGODB_PORT', '27017'))
MONGO_DB = 'reddit_data'
MONGO_COLLECTION = 'dataset'

PROMPT_TYPES = ('zero_shot', 'few_shot')
PROMPT_DIRS = {
    'zero_shot': os.path.join(EXPERIMENT_DIR, 'prompts', 'zero_shot_prompts'),
    'few_shot': os.path.join(EXPERIMENT_DIR, 'prompts', 'few_shot_prompts'),
}

# Domain (link_flair_text) to prompt file mapping
DOMAIN_TO_PROMPT = {
    'Books': 'books',
    'Film': 'movie_tv',
    'TV': 'movie_tv',
    'Music': 'music',
    'Games': 'games'
}

# What is sent to the model as the user's request.
#   title           -> submission title only (used for the test runs up to 2026-10-01)
#   title_selftext  -> title, followed by the submission body if it has one (default: same
#                      information the human commenters saw)
INPUT_MODES = ('title', 'title_selftext')
DEFAULT_INPUT_MODE = 'title_selftext'

# Only these fields are loaded from MongoDB (the embedded comments are not needed).
SUBMISSION_PROJECTION = {'_id': 1, 'id': 1, 'title': 1, 'selftext': 1, 'link_flair_text': 1, 'final_recommendations': 1}

# Ollama silently drops the beginning of prompts longer than num_ctx. prompt_eval_count close to
# the context window is used as an indicator that this happened.
PROMPT_TRUNCATION_THRESHOLD = 0.95

ollama_client = OllamaClient()
openai_client = OpenAiClient()


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_submission_text(submission, input_mode=DEFAULT_INPUT_MODE):
    """Returns the request text shown to the model for a submission."""
    title = (submission.get('title') or '').strip()
    if input_mode == 'title_selftext':
        selftext = (submission.get('selftext') or '').strip()
        if selftext and selftext not in ('[deleted]', '[removed]'):
            return f"{title}\n\n{selftext}"
    return title


_prompt_cache = {}


def load_prompt(prompt_type, domain):
    """Load (and cache) the prompt template for a prompt type and domain."""
    prompt_category = DOMAIN_TO_PROMPT.get(domain)
    if not prompt_category:
        raise ValueError(f"Unknown domain: {domain}")

    prompt_file = os.path.join(PROMPT_DIRS[prompt_type], f"{prompt_category}_{prompt_type}.txt")
    if prompt_file not in _prompt_cache:
        if not os.path.exists(prompt_file):
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
        with open(prompt_file, 'r', encoding='utf-8') as f:
            _prompt_cache[prompt_file] = f.read()
    return _prompt_cache[prompt_file]


class FewShotSampler:
    """Selects few-shot examples from the same domain.

    The selection is seeded per submission (seed + submission id), so
      - every model sees exactly the same examples for a given submission (fair comparison), and
      - the selection is reproducible across runs and independent of processing order / --limit.
    The pool is the whole dataset (not only the --limit subset); the submission itself is excluded.
    """

    def __init__(self, all_submissions, seed, num_examples=config.FEW_SHOT_NUM_EXAMPLES, input_mode=DEFAULT_INPUT_MODE):
        self.seed = seed
        self.num_examples = num_examples
        self.input_mode = input_mode
        self.pool = {}
        for sub in all_submissions:
            self.pool.setdefault(sub.get('link_flair_text'), []).append(sub)
        # sort so the selection does not depend on the order in which submissions were loaded
        for subs in self.pool.values():
            subs.sort(key=lambda s: str(s.get('id')))

    def sample(self, submission):
        domain = submission.get('link_flair_text')
        submission_id = submission.get('id')
        candidates = [s for s in self.pool.get(domain, []) if s.get('id') != submission_id]
        num_examples = min(self.num_examples, len(candidates))
        if num_examples < self.num_examples:
            debug_msg(f"Only {len(candidates)} few-shot examples available for domain {domain}", level=logging.WARNING)

        rng = random.Random(f"{self.seed}:{submission_id}")
        selected = rng.sample(candidates, num_examples)
        return [
            {
                'id': sub.get('id'),
                'submission_text': build_submission_text(sub, self.input_mode),
                'recommendations': (sub.get('final_recommendations') or [])[:config.FEW_SHOT_MAX_RECOMMENDATIONS_PER_EXAMPLE],
            }
            for sub in selected
        ]


def format_few_shot_prompt(prompt_template, submission_text, examples):
    """Insert examples (placeholders <random_example_N:...>) and the request text into the template."""
    prompt = prompt_template
    for i in range(1, config.FEW_SHOT_NUM_EXAMPLES + 1):
        placeholder = f'<random_example_{i}:submission_text_and_recommendations>'
        if i <= len(examples):
            example = examples[i - 1]
            recommendations_text = json.dumps(example['recommendations'], indent=2, ensure_ascii=False)
            text = f"Example {i}:\nRequest: {example['submission_text']}\nRecommendations: {recommendations_text}"
        else:
            text = ''
        prompt = prompt.replace(placeholder, text)
    return prompt.replace('<submission_text>', submission_text)


def format_zero_shot_prompt(prompt_template, submission_text):
    return prompt_template.replace('<submission_text>', submission_text)


# ---------------------------------------------------------------------------
# LLM calls
# ---------------------------------------------------------------------------

RETRYABLE_ERRORS = TRANSPORT_ERRORS + OPENAI_INFRASTRUCTURE_ERRORS + (ResponseError,)


def generate_recommendations(model, prompt):
    """Calls the model, retrying infrastructure errors. Raises InfrastructureError if they persist."""
    last_error = None
    for attempt in range(1, config.LLM_MAX_RETRIES + 1):
        try:
            if models_config.is_proprietary_model(model):
                return openai_client.generate(
                    model=models_config.get_model_version(model),
                    system_prompt='',  # Prompt is in user message
                    user_prompt=prompt
                )
            return ollama_client.generate(
                model=model,
                system_prompt='',  # Prompt is in user message
                user_prompt=prompt
            )
        except RETRYABLE_ERRORS as e:
            last_error = e
            wait = config.LLM_RETRY_BACKOFF_SECONDS * attempt
            debug_msg(f"LLM call failed (attempt {attempt}/{config.LLM_MAX_RETRIES}): "
                      f"{type(e).__name__}: {e}", level=logging.WARNING)
            if attempt < config.LLM_MAX_RETRIES:
                time.sleep(wait)
    raise InfrastructureError(f"{model}: giving up after {config.LLM_MAX_RETRIES} attempts: "
                              f"{type(last_error).__name__}: {last_error}")


def ensure_model_available(model):
    if models_config.is_proprietary_model(model):
        try:
            return openai_client.is_available(models_config.get_model_version(model))
        except OPENAI_INFRASTRUCTURE_ERRORS as e:
            raise InfrastructureError(f"OpenAI API not usable: {type(e).__name__}: {e}") from e
    try:
        return ollama_client.load_model(model)
    except TRANSPORT_ERRORS as e:
        raise InfrastructureError(f"Ollama server not reachable at {ollama_client.host}: {e}") from e


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _strip_code_fences(text):
    if text.startswith('```'):
        lines = text.split('\n')
        if len(lines) > 2:
            text = '\n'.join(lines[1:-1] if lines[-1].strip().startswith('```') else lines[1:])
    return text.strip()


def parse_recommendation_response(llm_response):
    """
    Parse an LLM response and extract the recommendations.

    valid_json:      the response is syntactically valid JSON
    valid_structure: the JSON follows the requested format ({"recommendations": [...]}, or a bare list)
    If the structure deviates (e.g. {"books": [...]}) the recommendations are still extracted
    leniently, but valid_structure is False so format adherence can be analysed separately.
    """
    result = {
        'valid_json': False,
        'valid_structure': False,
        'recommendations': [],
        'nr_recs': 0,
        'nr_recs_unique': 0,
        'raw_response': llm_response.message or '',
        'error': None
    }

    if not llm_response.ok:
        result['error'] = llm_response.error or llm_response.finish_reason or 'Unknown error'
        return result

    try:
        parsed = json.loads(_strip_code_fences((llm_response.message or '').strip()))
    except json.JSONDecodeError as e:
        result['error'] = f'JSON decode error: {e}'
        return result

    result['valid_json'] = True

    if isinstance(parsed, list):
        recommendations = parsed
        result['valid_structure'] = True
    elif isinstance(parsed, dict):
        if isinstance(parsed.get('recommendations'), list):
            recommendations = parsed['recommendations']
            result['valid_structure'] = True
        else:
            list_values = [v for v in parsed.values() if isinstance(v, list)]
            if len(list_values) == 1:
                recommendations = list_values[0]          # e.g. {"books": [...]}
            elif parsed:
                recommendations = [parsed]                # single recommendation object
            else:
                recommendations = []
    else:
        result['error'] = 'Unexpected JSON structure'
        return result

    recommendations = [rec for rec in recommendations if isinstance(rec, dict)]
    result['recommendations'] = recommendations
    result['nr_recs'] = len(recommendations)

    unique_titles = {
        str(rec['title']).lower().strip()
        for rec in recommendations
        if rec.get('title') not in (None, '')
    }
    result['nr_recs_unique'] = len(unique_titles)
    return result


def is_prompt_truncated(model, llm_response):
    if models_config.is_proprietary_model(model) or not config.CONTEXT_WINDOW:
        return False
    return (llm_response.prompt_token_count or 0) >= PROMPT_TRUNCATION_THRESHOLD * config.CONTEXT_WINDOW


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def process_submission(model, prompt_type, submission, input_mode, few_shot_sampler):
    """Runs one submission through one model/prompt type and returns the result row."""
    submission_id = submission.get('id')
    domain = submission.get('link_flair_text', '')
    ground_truth = submission.get('final_recommendations') or []
    submission_text = build_submission_text(submission, input_mode)

    row = {
        'submission_id': submission_id,
        'domain': domain,
        'model': model,
        'prompt_type': prompt_type,
        'input_mode': input_mode,
        'submission_text': submission_text,
        'nr_ground_truth': len(ground_truth),
        'ground_truth': json.dumps(ground_truth, ensure_ascii=False),
        'timestamp': datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
    }

    prompt_template = load_prompt(prompt_type, domain)
    if prompt_type == 'few_shot':
        examples = few_shot_sampler.sample(submission)
        row['few_shot_example_ids'] = json.dumps([e['id'] for e in examples])
        prompt = format_few_shot_prompt(prompt_template, submission_text, examples)
    else:
        prompt = format_zero_shot_prompt(prompt_template, submission_text)

    llm_response = generate_recommendations(model, prompt)  # may raise InfrastructureError
    parsed = parse_recommendation_response(llm_response)
    prompt_truncated = is_prompt_truncated(model, llm_response)
    if prompt_truncated:
        debug_msg(f"Prompt for {submission_id} probably truncated by the context window "
                  f"({llm_response.prompt_token_count} tokens)", level=logging.WARNING)

    row.update({
        'valid_json': parsed['valid_json'],
        'valid_structure': parsed['valid_structure'],
        'nr_recs': parsed['nr_recs'],
        'nr_recs_unique': parsed['nr_recs_unique'],
        'llm_recommendations': json.dumps(parsed['recommendations'], ensure_ascii=False),
        'raw_response': parsed['raw_response'],
        'finish_reason': llm_response.finish_reason or '',
        'response_truncated': llm_response.finish_reason == 'length',
        'prompt_truncated': prompt_truncated,
        'prompt_tokens': llm_response.prompt_token_count or 0,
        'response_tokens': llm_response.message_token_count or 0,
        'duration_seconds': llm_response.duration_seconds if llm_response.duration_seconds is not None else '',
        'error': parsed['error'] or '',
    })
    return row


def process_submissions(submissions, all_submissions, models, prompt_types, progress_file_path=None,
                        model_setup=None, output_root=None, input_mode=DEFAULT_INPUT_MODE, seed=config.SEED):
    """
    Process submissions for every model and prompt type. Returns an exit code.

    Results are appended row by row to <output_dir>/<model>/<prompt_type>_results.csv. Rows that
    already exist in that file are skipped, so re-running with the same --progress-file resumes.
    """
    run_config = {
        'inputMode': input_mode,
        'seed': seed,
        'ollamaOptions': OllamaClient.build_options(),
        'ollamaRequestTimeout': config.OLLAMA_REQUEST_TIMEOUT,
        'fewShotNumExamples': config.FEW_SHOT_NUM_EXAMPLES,
        'fewShotMaxRecommendationsPerExample': config.FEW_SHOT_MAX_RECOMMENDATIONS_PER_EXAMPLE,
        'openaiOptions': {'response_format': 'json_object', 'temperature': 'API default'},
        'nrSubmissions': len(submissions),
    }
    progress_data, progress_file_path = progress_utils.create_or_load_progress_structure(
        models, prompt_types, progress_file_path, model_setup, output_root=output_root, run_config=run_config
    )

    models = progress_utils.get_setup_property_by_key(progress_data, progress_utils.SETUP_MODELS_KEY)
    prompt_types = progress_utils.get_setup_property_by_key(progress_data, progress_utils.SETUP_SYSTEM_PROMPTS_KEY)
    output_directory = progress_utils.get_setup_property_by_key(progress_data, progress_utils.SETUP_OUTPUT_DIRECTORY)
    debug_msg(f"Progress file: {progress_file_path}", level=logging.INFO)
    create_directory_if_not_exists(output_directory)

    if any(not models_config.is_proprietary_model(m) for m in models):
        ollama_version = ollama_client.get_server_version()
        debug_msg(f"Ollama server version: {ollama_version}", level=logging.INFO)
        progress_utils.update_run_config(progress_file_path, {'ollamaServerVersion': ollama_version})

    few_shot_sampler = FewShotSampler(all_submissions, seed, input_mode=input_mode) if 'few_shot' in prompt_types else None
    exit_code = EXIT_OK

    for model_index, model in enumerate(models):
        if all(progress_utils.is_prompt_completed(progress_data, model_index, i) for i in range(len(prompt_types))):
            debug_msg(f"Skipping completed model: {model}", level=logging.INFO)
            continue

        if not ensure_model_available(model):  # may raise InfrastructureError
            debug_msg(f"Model {model} is not available, skipping.", level=logging.WARNING)
            for prompt_index in range(len(prompt_types)):
                progress_utils.set_model_prompt_not_available(model, prompt_index, progress_file_path)
            exit_code = EXIT_MODEL_NOT_AVAILABLE
            continue

        model_directory = os.path.join(output_directory, models_config.model_slug(model))
        create_directory_if_not_exists(model_directory)

        for prompt_index, prompt_type in enumerate(prompt_types):
            if progress_utils.is_prompt_completed(progress_data, model_index, prompt_index):
                debug_msg(f"Skipping completed: {model} with {prompt_type}", level=logging.INFO)
                continue

            progress_utils.set_model_prompt_running(model, prompt_index, progress_file_path)
            output_path = os.path.join(model_directory, f'{prompt_type}_results.csv')

            with ResultWriter(output_path) as writer:
                todo = [s for s in submissions if str(s.get('id')) not in writer.done_ids]
                debug_msg(f"Processing {model} with {prompt_type}: {len(todo)} to do, "
                          f"{len(submissions) - len(todo)} already done", level=logging.INFO)

                for submission in tqdm(todo, desc=f"{model} - {prompt_type}", mininterval=30):
                    if not submission.get('title') or not submission.get('link_flair_text'):
                        debug_msg(f"Skipping submission {submission.get('id')}: missing title or domain", level=logging.WARNING)
                        continue
                    try:
                        row = process_submission(model, prompt_type, submission, input_mode, few_shot_sampler)
                    except InfrastructureError:
                        progress_utils.set_model_prompt_failed(model, prompt_index, progress_file_path)
                        raise
                    writer.write(row)

            progress_utils.set_model_prompt_finished(model, prompt_index, progress_file_path)
            debug_msg(f"Finished {model} with {prompt_type}: {output_path}", level=logging.INFO)

    return exit_code


def load_submissions(collection, limit=None):
    """Loads all submissions in a deterministic order (by _id). Returns (selected, all)."""
    all_submissions = list(collection.find({}, SUBMISSION_PROJECTION).sort('_id', 1))
    selected = all_submissions[:limit] if limit else all_submissions
    return selected, all_submissions


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Generate recommendations using LLMs and store them with the ground truth.')
    parser.add_argument('--model-setup', type=str, choices=sorted(models_config.MODEL_SETUP_DICT),
                        help='Model setup key (see framework/models.py)')
    parser.add_argument('--model', type=str, nargs='+',
                        help='One or more model keys, e.g. llama3.2:3b-instruct-q4_K_M openai:gpt-5-mini')
    parser.add_argument('--prompt-types', type=str, nargs='+', choices=PROMPT_TYPES, default=list(PROMPT_TYPES),
                        help='Prompt types to test')
    parser.add_argument('--progress-file', type=str,
                        help='Path to a progress file. Created if it does not exist; resumed if it does.')
    parser.add_argument('--limit', type=int, help='Only use the first N submissions (for testing)')
    parser.add_argument('--output-dir', type=str, default=config.RESULTS_DATA_PATH,
                        help='Root output directory (used when no --progress-file is given)')
    parser.add_argument('--input-mode', type=str, choices=INPUT_MODES, default=DEFAULT_INPUT_MODE,
                        help='What is sent as the request: title only, or title + selftext')
    parser.add_argument('--seed', type=int, default=config.SEED,
                        help='Seed for Ollama sampling and few-shot example selection')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    start_time = datetime.now()
    debug_msg(f"Started {sys.argv[0]} with {argv if argv is not None else sys.argv[1:]}", level=logging.INFO)

    input_mode, seed = args.input_mode, args.seed

    # When resuming, the stored configuration wins so that a run never mixes settings.
    if args.progress_file and os.path.exists(args.progress_file):
        stored = progress_utils.get_setup_property_by_key(
            json_utils.read_json_file(args.progress_file), progress_utils.SETUP_RUN_CONFIG_KEY) or {}
        if stored:
            if (stored.get('inputMode'), stored.get('seed')) != (input_mode, seed):
                debug_msg(f"Resuming: using stored inputMode={stored.get('inputMode')} seed={stored.get('seed')} "
                          f"from the progress file (ignoring command line values)", level=logging.WARNING)
            input_mode = stored.get('inputMode', input_mode)
            seed = stored.get('seed', seed)
    config.SEED = seed

    # Determine models
    if args.model:
        model_list = args.model
    elif args.model_setup:
        model_list = models_config.get_models_for_setup(args.model_setup)
    else:
        model_list = models_config.get_all()
    unknown = [m for m in model_list if not models_config.is_known_model(m)]
    if unknown:
        debug_msg(f"Models not in framework/models.py (will still be tried): {unknown}", level=logging.WARNING)
    debug_msg(f"Models ({len(model_list)}): {model_list}", level=logging.INFO)

    # Connect to MongoDB
    try:
        client = MongoClient(MONGO_HOST, MONGO_PORT, serverSelectionTimeoutMS=30000)
        collection = client[MONGO_DB][MONGO_COLLECTION]
        submissions, all_submissions = load_submissions(collection, args.limit)
    except PyMongoError as e:
        debug_msg(f"MongoDB not reachable at {MONGO_HOST}:{MONGO_PORT}: {e}", level=logging.ERROR)
        return EXIT_CONFIG_ERROR
    if not submissions:
        debug_msg(f"No submissions found in {MONGO_DB}.{MONGO_COLLECTION}", level=logging.ERROR)
        return EXIT_CONFIG_ERROR
    debug_msg(f"MongoDB {MONGO_DB}.{MONGO_COLLECTION}: {len(all_submissions)} submissions, "
              f"using {len(submissions)}", level=logging.INFO)

    try:
        exit_code = process_submissions(
            submissions=submissions,
            all_submissions=all_submissions,
            models=model_list,
            prompt_types=args.prompt_types,
            progress_file_path=args.progress_file,
            model_setup=args.model_setup,
            output_root=args.output_dir,
            input_mode=input_mode,
            seed=seed,
        )
    except InfrastructureError as e:
        debug_msg(f"Aborting - infrastructure error: {e}. Re-run with the same --progress-file to resume.",
                  level=logging.ERROR)
        exit_code = EXIT_INFRASTRUCTURE_ERROR

    debug_msg(f"Finished {sys.argv[0]} - exit code {exit_code} - duration: {datetime.now() - start_time}", level=logging.INFO)
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
