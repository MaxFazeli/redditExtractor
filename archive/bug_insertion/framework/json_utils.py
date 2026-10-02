import json
import os
import jsonschema


def is_valid_json_string(json_string):
    """Check if the input string is a valid JSON."""
    try:
        data = json.loads(json_string)
        return True, data
    except ValueError:
        return False, None
    

def load_json_from_file(file_path):
    """Load JSON from a file."""
    with open(file_path, 'r') as file:
        return json.load(file)


def is_expected_json_structure(data, description=False, difference=False):
    schema = load_json_from_file("json_response_schema.json")
    try:
        jsonschema.validate(instance=data, schema=schema)
        return True
    except jsonschema.exceptions.ValidationError as err:
        print(f"JSON data is invalid: {err.message}")
        return False


def get_object_as_json(obj):
    return json.dumps(obj, indent=2) if obj is not None else None


def write_json(output_path, data):
    """Write the metrics dictionary to a JSON file."""
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write the dictionary to a JSON file
    with open(output_path, 'w') as outfile:
        json.dump(data, outfile, indent=4)


def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data