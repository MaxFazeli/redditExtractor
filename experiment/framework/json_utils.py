import json
import os


def is_valid_json_string(json_string):
    """Check if the input string is a valid JSON."""
    try:
        data = json.loads(json_string)
        return True, data
    except ValueError:
        return False, None


def load_json_from_file(file_path):
    """Load JSON from a file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def get_object_as_json(obj):
    return json.dumps(obj, indent=2, default=str) if obj is not None else None


def write_json(output_path, data):
    """Write data to a JSON file atomically (temp file + rename), so that a crash or
    container stop during the write can never leave a truncated progress file behind."""
    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    tmp_path = f"{output_path}.tmp"
    with open(tmp_path, 'w', encoding='utf-8') as outfile:
        json.dump(data, outfile, indent=4)
        outfile.flush()
        os.fsync(outfile.fileno())
    os.replace(tmp_path, output_path)


def read_json_file(file_path):
    return load_json_from_file(file_path)
