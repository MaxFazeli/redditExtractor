OPEN_SOURCE_MODEL_DICT = {
    # Tiny Models (< 4 billion parameters)
    "gemma:2b-instruct-v1.1-q4_0": {"parameters": "2b", "category": "tiny", "coding_optimized": False},
    "gemma2:2b-instruct-q4_0": {"parameters": "2b", "category": "tiny", "coding_optimized": False},
    "phi3:3.8b-mini-4k-instruct-q4_0": {"parameters": "3.8b", "category": "tiny", "coding_optimized": False},
    "phi3:3.8b-mini-128k-instruct-q4_0": {"parameters": "3.8b", "category": "tiny", "coding_optimized": False},
    "qwen2:0.5b-instruct-q4_0": {"parameters": "0.5b", "category": "tiny", "coding_optimized": False},
    "qwen2:1.5b-instruct-q4_0": {"parameters": "1.5b", "category": "tiny", "coding_optimized": False},
    "starcoder:1b-base-q4_0": {"parameters": "1b", "category": "tiny", "coding_optimized": True},
    "starcoder:3b-base-q4_0": {"parameters": "3b", "category": "tiny", "coding_optimized": True},
    "starcoder2:3b-q4_0": {"parameters": "3b", "category": "tiny", "coding_optimized": True},
    "deepseek-coder:1.3b-instruct-q4_0": {"parameters": "1.3b", "category": "tiny", "coding_optimized": True},
    "granite-code:3b-instruct-q4_0": {"parameters": "3b", "category": "tiny", "coding_optimized": True},
    "stable-code:3b-instruct-q4_0": {"parameters": "3b", "category": "tiny", "coding_optimized": True},
    "yi-coder:1.5b-base-q4_0": {"parameters": "1.5b", "category": "tiny", "coding_optimized": True},
    "deepseek-r1:1.5b-qwen-distill-q4_K_M": {"parameters": "1.5b", "category": "tiny", "coding_optimized": False},
    "llama3.2:1b-instruct-q4_0": {"parameters": "1b", "category": "tiny", "coding_optimized": False},

    # Small Models (≥ 4 and < 10 billion parameters)
    "gemma:7b-instruct-v1.1-q4_0": {"parameters": "7b", "category": "small", "coding_optimized": False},
    "gemma2:9b-instruct-q4_0": {"parameters": "9b", "category": "small", "coding_optimized": False},
    "llama3:8b-instruct-q4_0": {"parameters": "8b", "category": "small", "coding_optimized": False},
    "llama3.1:8b-instruct-q4_0": {"parameters": "8b", "category": "small", "coding_optimized": False},
    "mistral:7b-instruct-v0.3-q4_0": {"parameters": "7b", "category": "small", "coding_optimized": False},
    "mixtral:8x7b-instruct-v0.1-q4_0": {"parameters": "8x7b", "category": "small", "coding_optimized": False},
    "qwen2:7b-instruct-q4_0": {"parameters": "7b", "category": "small", "coding_optimized": False},
    "yi:6b-chat-v1.5-q4_0": {"parameters": "6b", "category": "small", "coding_optimized": False},
    "yi:9b-chat-v1.5-q4_0": {"parameters": "9b", "category": "small", "coding_optimized": False},
    "codellama:7b-instruct-q4_0": {"parameters": "7b", "category": "small", "coding_optimized": True},
    "deepseek-coder:6.7b-instruct-q4_0": {"parameters": "6.7b", "category": "small", "coding_optimized": True},
    "codegemma:7b-instruct-q4_0": {"parameters": "7b", "category": "small", "coding_optimized": True},
    "starcoder:7b-base-q4_0": {"parameters": "7b", "category": "small", "coding_optimized": True},
    "starcoder2:7b-instruct-q4_0": {"parameters": "7b", "category": "small", "coding_optimized": True},
    "qwen2.5-coder:7b-instruct-q4_0": {"parameters": "7b", "category": "small", "coding_optimized": True},
    "codegeex4:9b-all-q4_0": {"parameters": "9b", "category": "small", "coding_optimized": True},
    "granite-code:8b-instruct-q4_0": {"parameters": "8b", "category": "small", "coding_optimized": True},
    "codeqwen:7b-code-v1.5-q4_0": {"parameters": "7b", "category": "small", "coding_optimized": True},
    "wizardcoder:7b-python-q4_0": {"parameters": "7b", "category": "small", "coding_optimized": True},
    "dolphincoder:7b-starcoder2-q4_0": {"parameters": "7b", "category": "small", "coding_optimized": True},
    "yi-coder:9b-base-q4_0": {"parameters": "9b", "category": "small", "coding_optimized": True},
    "magicoder:7b-s-cl-q4_0": {"parameters": "7b", "category": "small", "coding_optimized": True},
    "deepseek-r1:7b-qwen-distill-q4_K_M": {"parameters": "7b", "category": "small", "coding_optimized": False},
    "deepseek-r1:8b-llama-distill-q4_K_M": {"parameters": "8b", "category": "small", "coding_optimized": False},
    "llama3.2:3b-instruct-q4_0": {"parameters": "3b", "category": "small", "coding_optimized": False},

    # Medium Models (≥ 10 and < 50 billion parameters)
    "gemma2:27b-instruct-q4_0": {"parameters": "27b", "category": "medium", "coding_optimized": False},
    "phi3:14b-medium-4k-instruct-q4_0": {"parameters": "14b", "category": "medium", "coding_optimized": False},
    "phi3:14b-medium-128k-instruct-q4_0": {"parameters": "14b", "category": "medium", "coding_optimized": False},
    "yi:34b-chat-v1.5-q4_0": {"parameters": "34b", "category": "medium", "coding_optimized": False},
    "codellama:13b-instruct-q4_0": {"parameters": "13b", "category": "medium", "coding_optimized": True},
    "codellama:34b-instruct-q4_0": {"parameters": "34b", "category": "medium", "coding_optimized": True},
    "deepseek-coder-v2:16b-lite-instruct-q4_0": {"parameters": "16b", "category": "medium", "coding_optimized": True},
    "starcoder:15b-base-q4_0": {"parameters": "15b", "category": "medium", "coding_optimized": True},
    "starcoder2:15b-instruct-q4_0": {"parameters": "15b", "category": "medium", "coding_optimized": True},
    "codestral:22b-v0.1-q4_0": {"parameters": "22b", "category": "medium", "coding_optimized": True},
    "codebooga:34b-v0.1-q4_0": {"parameters": "34b", "category": "medium", "coding_optimized": True},
    "granite-code:20b-instruct-q4_0": {"parameters": "20b", "category": "medium", "coding_optimized": True},
    "granite-code:34b-instruct-q4_0": {"parameters": "34b", "category": "medium", "coding_optimized": True},
    "wizardcoder:13b-python-q4_0": {"parameters": "13b", "category": "medium", "coding_optimized": True},
    "wizardcoder:33b-v1.1-q4_0": {"parameters": "33b", "category": "medium", "coding_optimized": True},
    "wizardcoder:34b-python-q4_0": {"parameters": "34b", "category": "medium", "coding_optimized": True},
    "dolphincoder:15b-starcoder2-q4_0": {"parameters": "15b", "category": "medium", "coding_optimized": True},
    "codeup:13b-llama2-chat-q4_0": {"parameters": "13b", "category": "medium", "coding_optimized": True},
    "deepseek-r1:14b-qwen-distill-q4_K_M": {"parameters": "14b", "category": "medium", "coding_optimized": False},
    "deepseek-r1:32b-qwen-distill-q4_K_M": {"parameters": "32b", "category": "medium", "coding_optimized": False},
    "phi4:14b-q4_K_M": {"parameters": "14b", "category": "medium", "coding_optimized": False},

    # Large Models (≥ 50 billion parameters)
    "llama3:70b-instruct-q4_0": {"parameters": "70b", "category": "large", "coding_optimized": False},
    "llama3.1:70b-instruct-q4_0": {"parameters": "70b", "category": "large", "coding_optimized": False},
    "qwen2:72b-instruct-q4_0": {"parameters": "72b", "category": "large", "coding_optimized": False},
    "mistral-large:123b-instruct-2407-q4_0": {"parameters": "123b", "category": "large", "coding_optimized": False},
    #"llama3.1:405b-instruct-q4_0": {"parameters": "405b", "category": "large", "coding_optimized": False},
    #"deepseek-coder-v2:236b-instruct-q4_0": {"parameters": "236b", "category": "large", "coding_optimized": True},
    "deepseek-r1:70b-llama-distill-q4_K_M": {"parameters": "70b", "category": "large", "coding_optimized": False},
    "llama3.3:70b-instruct-q4_0": {"parameters": "70b", "category": "large", "coding_optimized": False}
}

PROPRIETARY_MODEL_DICT = {
    "openai:gpt-4o": {"parameters": None, "category": "large", "coding_optimized": False},
}


def get_models_by_category(category):
    """Returns a list of models for the specified category."""
    return [model for model, details in OPEN_SOURCE_MODEL_DICT.items() if details["category"] == category]


def get_coding_optimized():
    """Returns a list of models optimized for coding."""
    return [model for model, details in OPEN_SOURCE_MODEL_DICT.items() if details["coding_optimized"]]


def get_all():
    """Returns a list of all model names."""
    return list(OPEN_SOURCE_MODEL_DICT.keys())


def get_category_by_model_key(model_key):
    """Returns the category of the given model key."""
    return OPEN_SOURCE_MODEL_DICT.get(model_key, {}).get("category")


def get_model_group(model_name):
    """Returns a list of models that belong to a specific group, followed by a colon."""
    target_prefix = model_name + ":"
    return [model for model in OPEN_SOURCE_MODEL_DICT.keys() if model.startswith(target_prefix)]


def get_models_with_all_category_variations():
    """Returns a list of model groups that have variations in all categories: tiny, small, medium, large."""

    # Create a dictionary to track the presence of categories for each model group
    group_category_tracker = {}

    for model, details in OPEN_SOURCE_MODEL_DICT.items():
        # Extract the model group name, which is everything before the first colon
        group_name = model.split(":")[0]
        category = details["category"]

        # Initialize the group's category list if it doesn't exist
        if group_name not in group_category_tracker:
            group_category_tracker[group_name] = set()

        # Add the current model's category to the group's set
        group_category_tracker[group_name].add(category)

    # Find all groups that have models in all four categories
    models_with_all_categories = [
        group for group, categories in group_category_tracker.items() if len(categories) == 4
    ]

    return models_with_all_categories


def is_proprietary_model(model_key):
    """Checks if the given model key is present in the proprietary model dictionary."""
    return model_key in PROPRIETARY_MODEL_DICT.keys()


def get_model_version(model_key):
    """Returns the part of the model key that comes after the colon."""
    # Split the key at the colon and return the part after it
    if ":" in model_key:
        return model_key.split(":", 1)[1]
    else:
        return None  # Return None if there is no colon


def parameter_to_float(parameter_str):
    """Converts a parameter string like '3.8b' or '8x7b' into a float representing its numerical value."""
    # Remove the 'b' character
    parameter_without_b = parameter_str.replace('b', '')

    if 'x' in parameter_without_b:
        # Split the string into its multiplicative components
        parts = parameter_without_b.split('x')
        # Convert them to floats and multiply
        result = float(parts[0]) * float(parts[1])
    else:
        # Directly convert the cleaned string to float
        result = float(parameter_without_b)

    return result


def get_models_smaller_than(max_parameters):
    """Returns a list of models that are smaller than the specified parameter count."""
    max_value = parameter_to_float(max_parameters)
    smaller_models = [
        model for model, details in OPEN_SOURCE_MODEL_DICT.items()
        if parameter_to_float(details["parameters"]) < max_value
    ]
    return smaller_models


def get_models_larger_than(min_parameters):
    """Returns a list of models that are larger than the specified parameter count."""
    min_value = parameter_to_float(min_parameters)
    larger_models = [
        model for model, details in OPEN_SOURCE_MODEL_DICT.items()
        if parameter_to_float(details["parameters"]) > min_value
    ]
    return larger_models


# Define the model setup dictionary
MODEL_SETUP_DICT = {
    "tiny_models": lambda: get_models_by_category("tiny"),
    "small_models": lambda: get_models_by_category("small"),
    "medium_models": lambda: get_models_by_category("medium"),
    "large_models": lambda: get_models_by_category("large"),
    "coding_optimized_models": lambda: get_coding_optimized(),
    "all_models": lambda: get_all(),
    "gemma2_models": lambda: get_model_group("gemma2"),
    "complete_models": lambda: get_models_with_all_category_variations(),
    "smaller_than_70b": lambda: get_models_smaller_than("70b"),
}

# Example of how to use the MODEL_SETUP_DICT to retrieve models
def get_models_for_setup(setup_name):
    if setup_name in MODEL_SETUP_DICT:
        return MODEL_SETUP_DICT[setup_name]()
    else:
        print("Invalid setup name.")
        return []


def main():
    # Get all models by category
    tiny_models = get_models_by_category("tiny")
    small_models = get_models_by_category("small")
    medium_models = get_models_by_category("medium")
    large_models = get_models_by_category("large")

    # Print models by category
    print("Tiny Models:", tiny_models)
    print("Small Models:", small_models)
    print("Medium Models:", medium_models)
    print("Large Models:", large_models)

    # Get and print coding optimized models
    coding_optimized_models = get_coding_optimized()
    print("Coding Optimized Models:", coding_optimized_models)

    # Get and print all models
    all_models = get_all()
    print("All Models:", all_models)

    # Example usage: Get models for group "gemma2"
    gemma2_models = get_model_group("gemma2")
    print("Gemma2 Models:", gemma2_models)

    # Get all model groups with variations in all categories
    complete_models = get_models_with_all_category_variations()
    print("Models with variations in all categories:", complete_models)

    # Check if a specific model key is proprietary
    full_model_key = "openai:gpt-4o"  # Example full model key
    is_proprietary = is_proprietary_model(full_model_key)
    print(f"Is '{full_model_key}' a proprietary model? {is_proprietary}")

    # Example usage: Get models smaller than 20 billion parameters
    smaller_than_20b = get_models_smaller_than("20b")
    print("Models smaller than 20b:", smaller_than_20b)

    # Example usage: Get models larger than 20 billion parameters
    larger_than_20b = get_models_larger_than("20b")
    print("Models larger than 20b:", larger_than_20b)


if __name__ == '__main__':
    main()