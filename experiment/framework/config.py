import os

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
# Root directory for experiment output. Overridden by --output-dir.
RESULTS_DATA_PATH = os.environ.get("RESULTS_DIR", "recommendation_results")

# ---------------------------------------------------------------------------
# Generation options (Ollama)
# ---------------------------------------------------------------------------
# Upper bound for generated tokens, identical for all models. For reasoning models
# (gpt-oss, deepseek-r1, qwen3) the internal "thinking" tokens count towards this limit:
# with the original 1000, gpt-oss:20b used up the whole budget while thinking and returned
# empty or cut-off JSON (smoke test 2026-10-01). Non-reasoning models stop on their own far
# earlier (<= ~620 tokens), so the higher limit does not change their output.
# Responses that still hit the limit are flagged (finish_reason == "length").
MAX_OUTPUT_TOKENS = 4096

# CPU threads used by Ollama for layers that are not offloaded to the GPU.
NUM_THREAD = 2

# Context window. Ollama silently truncates prompts that exceed num_ctx by dropping
# the *beginning* of the prompt (i.e. the instructions). 8192 leaves room for
# few-shot prompts; truncation is detected and flagged per row (prompt_truncated).
CONTEXT_WINDOW = 8192

# Sampling: temperature/top_p/top_k are left at each model's default (None = not sent),
# but a fixed seed is used so that runs are repeatable on the same hardware and
# Ollama version.
TEMPERATURE = None
SEED = 42

# Timeout (seconds) for a single HTTP request to Ollama. The first request after a
# model switch includes loading the model into GPU memory (minutes for 70B+ models).
OLLAMA_REQUEST_TIMEOUT = 1800

# Retries for transient infrastructure errors (connection reset, tunnel restart, ...).
LLM_MAX_RETRIES = 3
LLM_RETRY_BACKOFF_SECONDS = 30

# ---------------------------------------------------------------------------
# Few-shot prompting
# ---------------------------------------------------------------------------
FEW_SHOT_NUM_EXAMPLES = 3
FEW_SHOT_MAX_RECOMMENDATIONS_PER_EXAMPLE = 10
