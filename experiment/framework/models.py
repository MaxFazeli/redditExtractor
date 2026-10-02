"""
Model registry for the recommendation experiment.

Single source of truth for which models are evaluated. Ollama models are served on the
nvcluster GPU node; proprietary models are called through the OpenAI API directly from
the container.

Proprietary model keys use the prefix "openai:"; the part after the colon is the model
id sent to the API (e.g. "openai:gpt-5-mini" -> "gpt-5-mini").
"""

# "generation": 1 = original model set (tested May 2026), 2 = newer models added on 2026-10-02
# (successors / additional families: qwen3.5 across all sizes, gemma4, fully open olmo-3).
# "reasoning": model generates internal "thinking" tokens (counted towards num_predict).
OPEN_SOURCE_MODEL_DICT = {
    # Tiny (< 4B parameters)
    "llama3.2:3b-instruct-q4_K_M":             {"parameters": "3b",   "category": "tiny",   "reasoning": False, "generation": 1},

    # Small (4B - 10B parameters)
    "gemma3:4b-it-q4_K_M":                     {"parameters": "4b",   "category": "small",  "reasoning": False, "generation": 1},
    "llama3.1:8b-instruct-q4_K_M":             {"parameters": "8b",   "category": "small",  "reasoning": False, "generation": 1},
    "deepseek-r1:8b-0528-qwen3-q4_K_M":        {"parameters": "8b",   "category": "small",  "reasoning": True,  "generation": 1},
    "qwen3.5:4b-q4_K_M":                       {"parameters": "4b",   "category": "small",  "reasoning": True,  "generation": 2},
    "gemma4:e4b-it-q4_K_M":                    {"parameters": "4b",   "category": "small",  "reasoning": True,  "generation": 2},  # "effective 4B"
    "olmo-3:7b-instruct-q4_K_M":               {"parameters": "7b",   "category": "small",  "reasoning": False, "generation": 2},
    "qwen3.5:9b-q4_K_M":                       {"parameters": "9b",   "category": "small",  "reasoning": True,  "generation": 2},

    # Medium (10B - 50B parameters)
    "gpt-oss:20b":                             {"parameters": "20b",  "category": "medium", "reasoning": True,  "generation": 1},
    "mistral-small3.2:24b-instruct-2506-q4_K_M": {"parameters": "24b", "category": "medium", "reasoning": False, "generation": 1},
    "gemma3:27b-it-q4_K_M":                    {"parameters": "27b",  "category": "medium", "reasoning": False, "generation": 1},
    "qwen3:32b-q4_K_M":                        {"parameters": "32b",  "category": "medium", "reasoning": True,  "generation": 1},
    "deepseek-r1:32b-qwen-distill-q4_K_M":     {"parameters": "32b",  "category": "medium", "reasoning": True,  "generation": 1},
    "qwen3.5:27b-q4_K_M":                      {"parameters": "27b",  "category": "medium", "reasoning": True,  "generation": 2},
    "gemma4:31b-it-q4_K_M":                    {"parameters": "31b",  "category": "medium", "reasoning": True,  "generation": 2},
    "olmo-3.1:32b-instruct-q4_K_M":            {"parameters": "32b",  "category": "medium", "reasoning": False, "generation": 2},

    # Large (>= 50B parameters)
    "llama3.3:70b-instruct-q4_K_M":            {"parameters": "70b",  "category": "large",  "reasoning": False, "generation": 1},
    "gpt-oss:120b":                            {"parameters": "120b", "category": "large",  "reasoning": True,  "generation": 1},
    "mistral-large:123b-instruct-2411-q4_K_M": {"parameters": "123b", "category": "large",  "reasoning": False, "generation": 1},
    "qwen3.5:122b-a10b-q4_K_M":                {"parameters": "122b", "category": "large",  "reasoning": True,  "generation": 2},  # MoE, 10B active, 81 GB
}

# Parameter counts of proprietary models are not published; the category reflects the
# size tier the model is compared against in the thesis.
PROPRIETARY_MODEL_DICT = {
    "openai:gpt-5-nano": {"parameters": None, "category": "tiny",   "reasoning": True},
    "openai:gpt-5-mini": {"parameters": None, "category": "medium", "reasoning": True},
    "openai:gpt-5.2":    {"parameters": None, "category": "large",  "reasoning": True},
}

ALL_MODEL_DICT = {**OPEN_SOURCE_MODEL_DICT, **PROPRIETARY_MODEL_DICT}

CATEGORY_ORDER = ["tiny", "small", "medium", "large"]


def _sorted_by_category(model_dict):
    """Returns model keys ordered by size category, keeping insertion order within a category."""
    return [
        model
        for category in CATEGORY_ORDER
        for model, details in model_dict.items()
        if details["category"] == category
    ]


def get_models_by_category(category, include_proprietary=True):
    """Returns all models of the given size category."""
    source = ALL_MODEL_DICT if include_proprietary else OPEN_SOURCE_MODEL_DICT
    return [model for model, details in source.items() if details["category"] == category]


def get_ollama_models():
    return _sorted_by_category(OPEN_SOURCE_MODEL_DICT)


def get_proprietary_models():
    return _sorted_by_category(PROPRIETARY_MODEL_DICT)


def get_all():
    """Returns all models (Ollama + proprietary), ordered by size category."""
    return _sorted_by_category(ALL_MODEL_DICT)


def is_known_model(model_key):
    return model_key in ALL_MODEL_DICT


def is_proprietary_model(model_key):
    """Checks if the given model key refers to a proprietary (API) model."""
    return model_key in PROPRIETARY_MODEL_DICT or model_key.startswith("openai:")


def get_model_version(model_key):
    """Returns the API model id of a proprietary model key ("openai:gpt-5-mini" -> "gpt-5-mini")."""
    if ":" in model_key:
        return model_key.split(":", 1)[1]
    return model_key


def model_slug(model_key):
    """File-system safe name of a model key (used for directories and file names)."""
    return model_key.replace(":", "_").replace("/", "_")


MODEL_SETUP_DICT = {
    "tiny_models": lambda: get_models_by_category("tiny"),
    "small_models": lambda: get_models_by_category("small"),
    "medium_models": lambda: get_models_by_category("medium"),
    "large_models": lambda: get_models_by_category("large"),
    # all models that fit on a single 48 GB GPU - used for the first, low-footprint batch on the shared cluster
    "tiny_and_small_models": lambda: get_models_by_category("tiny", include_proprietary=False) + get_models_by_category("small", include_proprietary=False),
    "ollama_models": get_ollama_models,
    "openai_models": get_proprietary_models,
    "all_models": get_all,
}


def get_models_for_setup(setup_name):
    """Returns the model list for a setup key; raises ValueError for unknown keys."""
    if setup_name not in MODEL_SETUP_DICT:
        raise ValueError(
            f"Unknown model setup '{setup_name}'. Valid setups: {', '.join(MODEL_SETUP_DICT)}"
        )
    return MODEL_SETUP_DICT[setup_name]()


if __name__ == '__main__':
    for setup in MODEL_SETUP_DICT:
        print(f"{setup}: {get_models_for_setup(setup)}")
