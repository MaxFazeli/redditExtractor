import os
import json_utils
import config
import uuid

from datetime import datetime


SETUP_KEY = "setup"
SETUP_SYSTEM_PROMPTS_KEY = "systemPrompts"
SETUP_MODELS_KEY = "models"
SETUP_MODEL_SETUP_KEY = "modelSetup"
SETUP_CREATION_TIMESTAMP_KEY = "creationTimestamp"
SETUP_END_TIMESTAMP_KEY = "endTimestamp"
SETUP_TOTAL_MODELS_KEY = "totalModels"
PROGRESS_ENTRIES_KEY = "progress_entries"
SETUP_OUTPUT_DIRECTORY = "outputDirectory"


def create_or_load_progress_structure(models, system_prompts, progress_file_path=None, model_setup=None):
    """Load existing progress or create new if the file does not exist, defaulting to creating new if path is None."""
    if progress_file_path is None:
        creation_timestamp = datetime.now().strftime('%Y_%m_%d_%H%M%S')
        unique_id = str(uuid.uuid4())
        model_setup_path = f"_{model_setup}" if model_setup else ""
        progress_file_path = os.path.join(config.BUG_INSERSION_DATA_PATH, f"{creation_timestamp}{model_setup_path}_{unique_id}", "progress.json")
        create_initial_progress_structure(models, system_prompts, progress_file_path, model_setup=model_setup)
    elif not os.path.exists(progress_file_path):
        create_initial_progress_structure(models, system_prompts, progress_file_path, model_setup=model_setup)

    return json_utils.read_json_file(progress_file_path), progress_file_path


def create_initial_progress_structure(models, system_prompts, progress_file_path, model_setup=None):
    """Initialize the progress file with entries for all models and prompts."""
    setup = {
        SETUP_CREATION_TIMESTAMP_KEY: datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ'),
        SETUP_TOTAL_MODELS_KEY: len(models),
        SETUP_MODELS_KEY: models,
        SETUP_SYSTEM_PROMPTS_KEY: system_prompts,
        SETUP_MODEL_SETUP_KEY: model_setup,
        SETUP_END_TIMESTAMP_KEY: None,
        SETUP_OUTPUT_DIRECTORY: os.path.dirname(progress_file_path)
    }

    progress_entries = []

    for model_index, model in enumerate(models):
        prompts_status = {str(i): "not started" for i in range(len(system_prompts))}
        model_entry = {
            "index": model_index,
            "model": model,
            "prompts": prompts_status,
            "status": "not started",
            "startTimestamp": None,
            "endTimestamp": None
        }

        progress_entries.append(model_entry)

    progress_structure = {
        SETUP_KEY: setup,
        PROGRESS_ENTRIES_KEY: progress_entries
    }

    json_utils.write_json(progress_file_path, progress_structure)


def get_setup_properties(progress_structure):
    """Extract and return all setup properties from the given progress structure."""
    setup_properties = progress_structure.get(SETUP_KEY, {})
    return setup_properties


def get_setup_property_by_key(progress_structure, key):
    """Return the specified setup property by key from the given progress structure."""
    setup_properties = progress_structure.get(SETUP_KEY, {})
    return setup_properties.get(key, None)


def is_prompt_completed(progress_structure, model_index, prompt_index):
    return progress_structure.get(PROGRESS_ENTRIES_KEY)[model_index]['prompts'][str(prompt_index)] == 'finished'


def read_progress(progress_file_path):
    """Read the existing progress data from the file."""
    progress_data = json_utils.read_json_file(progress_file_path)
    return progress_data.get(PROGRESS_ENTRIES_KEY, [])


def set_model_prompt_running(model, prompt_index, progress_file_path):
    update_progress_for_model(model, prompt_index, 'running', progress_file_path)


def set_model_prompt_finished(model, prompt_index, progress_file_path):
    update_progress_for_model(model, prompt_index, 'finished', progress_file_path)


def set_model_prompt_not_available(model, prompt_index, progress_file_path):
    update_progress_for_model(model, prompt_index, 'not available', progress_file_path)


def update_progress_for_model(model, prompt_index, new_status, progress_file_path):
    """Update the progress for a specific prompt of a model in the progress.json file."""
    progress_data = json_utils.read_json_file(progress_file_path)
    progress_entries = progress_data.get(PROGRESS_ENTRIES_KEY, [])

    model_entry = next((entry for entry in progress_entries if entry['model'] == model), None)

    if model_entry:
        # Update prompt status
        model_entry['prompts'][str(prompt_index)] = new_status

        # Determine overall model status based on prompt statuses
        prompt_statuses = model_entry['prompts'].values()

        if all(status == "not started" for status in prompt_statuses):
            model_entry['status'] = "not started"
        elif all(status == "finished" for status in prompt_statuses):
            model_entry['status'] = "finished"
            model_entry['endTimestamp'] = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
        elif "running" in prompt_statuses:
            model_entry['status'] = "running"
            model_entry['startTimestamp'] = model_entry['startTimestamp'] or datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')

    # Check if all model entries are finished
    if all(entry['status'] == "finished" for entry in progress_entries):
        progress_data[SETUP_KEY]['endTimestamp'] = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')

    # Write changes back to the progress.json file
    json_utils.write_json(progress_file_path, progress_data)