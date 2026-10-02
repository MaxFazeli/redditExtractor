import os
import time

import openai
from openai import OpenAI

from llm_response import LlmResponse


def clean_response(response):
    """Extracts the response content and strips markdown code fences. Returns '' if there is no content."""
    try:
        content = response.choices[0].message.content
    except (IndexError, AttributeError):
        return ''
    if content is None:  # e.g. refusal, or all tokens spent on reasoning
        return ''
    return content.strip().replace('```json', '').replace('```', '').strip()


# Errors that say nothing about the model's answer -> must not be stored as a result.
INFRASTRUCTURE_ERRORS = (
    openai.APIConnectionError,
    openai.RateLimitError,
    openai.InternalServerError,
    openai.AuthenticationError,
    openai.PermissionDeniedError,
)


class OpenAiClient:
    def __init__(self):
        self.api_key = os.environ.get("OPENAI_API_KEY")
        # The SDK retries connection errors, 429 and 5xx with exponential backoff.
        self.client = OpenAI(api_key=self.api_key, max_retries=5, timeout=600) if self.api_key else None

    def is_available(self, model):
        """Checks that an API key is configured and the model id exists for this key."""
        if not self.client:
            print("OPENAI_API_KEY is not set - OpenAI models are not available.")
            return False
        try:
            self.client.models.retrieve(model)
            return True
        except openai.NotFoundError:
            print(f"OpenAI model '{model}' does not exist or is not accessible with this API key.")
            return False

    def generate(self, model, system_prompt, user_prompt):
        """Generates a response from the model.

        Notes:
            - JSON mode (response_format=json_object) mirrors Ollama's format="json".
            - No temperature/seed is sent: GPT-5 family (reasoning) models only accept the default
              temperature, so all OpenAI calls use the API defaults.
            - Infrastructure errors (see INFRASTRUCTURE_ERRORS) are raised to the caller.
        """
        if not self.client:
            return LlmResponse(ok=False, finish_reason="client_not_available",
                               error="OPENAI_API_KEY is not set")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        start = time.monotonic()
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
            )
        except INFRASTRUCTURE_ERRORS:
            raise
        except openai.APIStatusError as e:
            # e.g. 400 Bad Request caused by this particular prompt -> recorded as a failed row
            return LlmResponse(ok=False, finish_reason="api_error", error=f"{type(e).__name__}: {e}")

        return LlmResponse(
            ok=True,
            finish_reason=response.choices[0].finish_reason,
            message=clean_response(response),
            message_token_count=response.usage.completion_tokens,
            prompt_token_count=response.usage.prompt_tokens,
            duration_seconds=round(time.monotonic() - start, 3),
            llm_full_response=response,
            llm_full_response_json=response.model_dump_json(),
        )
