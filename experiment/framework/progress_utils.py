import os
import uuid

from datetime import datetime

import config
import json_utils


SETUP_KEY = "setup"
SETUP_SYSTEM_PROMPTS_KEY = "systemPrompts"   # list of prompt types (zero_shot, few_shot)
SETUP_MODELS_KEY = "models"
SETUP_MODEL_SETUP_KEY = "modelSetup"
SETUP_CREATION_TIMESTAMP_KEY = "creationTimestamp"
SETUP_END_TIMESTAMP_KEY = "endTimestamp"
SETUP_TOTAL_MODELS_KEY = "totalModels"
SETUP_OUTPUT_DIRECTORY = "outputDirectory"
SETUP_RUN_CONFIG_KEY = "runConfig"           # generation options, seed, input mode, ...
PROGRESS_ENTRIES_KEY = "progress_entries"

STATUS_NOT_STARTED = "not started"
STATUS_RUNNING = "running"
STATUS_FINISHED = "finished"
STATUS_FAILED = "failed"
STATUS_NOT_AVAILABLE = "not available"


def _now():
    return datetime.now().strftime('%Y-%m-%dT%H:%M:%S')


def create_or_load_progress_structure(models, system_prompts, progress_file_path=None, model_setup=None,
                                      output_root=None, run_config=None):
    """Load existing progress, or create a new progress file.

    If no path is given, a new run directory <output_root>/<timestamp>[_<setup>]_<uuid>/ is created.
    If a path is given that does not exist yet, the progress file is created there (this is how
    run_models.py gets stable, resumable paths).
    """
    if progress_file_path is None:
        output_root = output_root or config.RESULTS_DATA_PATH
        creation_timestamp = datetime.now().strftime('%Y_%m_%d_%H%M%S')
        unique_id = str(uuid.uuid4())
        model_setup_path = f"_{model_setup}" if model_setup else ""
        progress_file_path = os.path.join(output_root, f"{creation_timestamp}{model_setup_path}_{unique_id}", "progress.json")
        create_initial_progress_structure(models, system_prompts, progress_file_path, model_setup, run_config)
    elif not os.path.exists(progress_file_path):
        create_initial_progress_structure(models, system_prompts, progress_file_path, model_setup, run_config)

    return json_utils.read_json_file(progress_file_path), progress_file_path


def create_initial_progress_structure(models, system_prompts, progress_file_path, model_setup=None, run_config=None):
    """Initialize the progress file with entries for all models and prompts."""
    setup = {
        SETUP_CREATION_TIMESTAMP_KEY: _now(),
        SETUP_TOTAL_MODELS_KEY: len(models),
        SETUP_MODELS_KEY: models,
        SETUP_SYSTEM_PROMPTS_KEY: system_prompts,
        SETUP_MODEL_SETUP_KEY: model_setup,
        SETUP_END_TIMESTAMP_KEY: None,
        SETUP_OUTPUT_DIRECTORY: os.path.dirname(progress_file_path),
        SETUP_RUN_CONFIG_KEY: run_config or {},
    }

    progress_entries = []
    for model_index, model in enumerate(models):
        progress_entries.append({
            "index": model_index,
            "model": model,
            "prompts": {str(i): STATUS_NOT_STARTED for i in range(len(system_prompts))},
            "status": STATUS_NOT_STARTED,
            "startTimestamp": None,
            "endTimestamp": None,
        })

    json_utils.write_json(progress_file_path, {SETUP_KEY: setup, PROGRESS_ENTRIES_KEY: progress_entries})


def get_setup_properties(progress_structure):
    return progress_structure.get(SETUP_KEY, {})


def get_setup_property_by_key(progress_structure, key):
    return progress_structure.get(SETUP_KEY, {}).get(key, None)


def is_prompt_completed(progress_structure, model_index, prompt_index):
    return progress_structure.get(PROGRESS_ENTRIES_KEY)[model_index]['prompts'][str(prompt_index)] == STATUS_FINISHED


def read_progress(progress_file_path):
    return json_utils.read_json_file(progress_file_path).get(PROGRESS_ENTRIES_KEY, [])


def set_model_prompt_running(model, prompt_index, progress_file_path):
    update_progress_for_model(model, prompt_index, STATUS_RUNNING, progress_file_path)


def set_model_prompt_finished(model, prompt_index, progress_file_path):
    update_progress_for_model(model, prompt_index, STATUS_FINISHED, progress_file_path)


def set_model_prompt_failed(model, prompt_index, progress_file_path):
    update_progress_for_model(model, prompt_index, STATUS_FAILED, progress_file_path)


def set_model_prompt_not_available(model, prompt_index, progress_file_path):
    update_progress_for_model(model, prompt_index, STATUS_NOT_AVAILABLE, progress_file_path)


def update_run_config(progress_file_path, values):
    """Merge values into setup.runConfig (e.g. the Ollama server version observed during a run)."""
    progress_data = json_utils.read_json_file(progress_file_path)
    run_config = progress_data[SETUP_KEY].setdefault(SETUP_RUN_CONFIG_KEY, {})
    run_config.update(values)
    json_utils.write_json(progress_file_path, progress_data)


def update_progress_for_model(model, prompt_index, new_status, progress_file_path):
    """Update the status of one prompt type of a model and derive the overall model/run status."""
    progress_data = json_utils.read_json_file(progress_file_path)
    progress_entries = progress_data.get(PROGRESS_ENTRIES_KEY, [])

    model_entry = next((entry for entry in progress_entries if entry['model'] == model), None)

    if model_entry:
        model_entry['prompts'][str(prompt_index)] = new_status
        statuses = list(model_entry['prompts'].values())

        if new_status == STATUS_RUNNING:
            model_entry['startTimestamp'] = model_entry['startTimestamp'] or _now()

        if all(s == STATUS_NOT_STARTED for s in statuses):
            model_entry['status'] = STATUS_NOT_STARTED
        elif all(s == STATUS_FINISHED for s in statuses):
            model_entry['status'] = STATUS_FINISHED
            model_entry['endTimestamp'] = _now()
        elif STATUS_RUNNING in statuses:
            model_entry['status'] = STATUS_RUNNING
        elif STATUS_FAILED in statuses:
            model_entry['status'] = STATUS_FAILED
        elif STATUS_NOT_AVAILABLE in statuses:
            model_entry['status'] = STATUS_NOT_AVAILABLE
        else:
            model_entry['status'] = STATUS_RUNNING  # partially finished, rest not started

    if progress_entries and all(entry['status'] == STATUS_FINISHED for entry in progress_entries):
        progress_data[SETUP_KEY][SETUP_END_TIMESTAMP_KEY] = _now()

    json_utils.write_json(progress_file_path, progress_data)
