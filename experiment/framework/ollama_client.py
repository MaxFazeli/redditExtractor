import json
import os
import urllib.request

import httpx
from ollama import Client
from ollama._types import ResponseError

import config
import json_utils
from llm_response import LlmResponse


class InfrastructureError(RuntimeError):
    """Raised when the Ollama server cannot be reached or keeps failing (tunnel down, job ended, OOM, ...).

    Such errors say nothing about the model's answer, so the affected submission must not be
    recorded as a result; the run is aborted and can be resumed later.
    """


# Errors raised by httpx when the connection (SSH tunnel -> GPU node) is broken or times out.
TRANSPORT_ERRORS = (httpx.TransportError,)


class OllamaClient:
    def __init__(self, host=None, port=None):
        if host is None:
            self.host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
        else:
            self.host = f"{host}:{port}"
        if not self.host.startswith('http'):
            self.host = f"http://{self.host}"
        self.client = Client(host=self.host, timeout=config.OLLAMA_REQUEST_TIMEOUT)

    def generate(self, model, system_prompt, user_prompt):
        """Generates a response from the specified model.

        Raises:
            ResponseError: the server answered with an error status (e.g. model failed to load).
            httpx.TransportError: the server could not be reached.
        """
        options = self.build_options()

        response = self.client.generate(
            model=model,
            system=system_prompt,
            prompt=user_prompt,
            format="json",
            options=options,
            stream=False
        )

        total_duration_ns = response.get("total_duration") or 0
        return LlmResponse(
            ok=bool(response.get("done")),
            finish_reason=response.get("done_reason"),
            message=str(response.get("response", "")).strip(),
            message_token_count=response.get("eval_count", 0),
            prompt_token_count=response.get("prompt_eval_count", 0),
            duration_seconds=round(total_duration_ns / 1e9, 3),
            llm_full_response=response,
            llm_full_response_json=json_utils.get_object_as_json(response),
        )

    def load_model(self, model):
        """Makes sure the model is available on the server, pulling it if necessary.

        Returns True if the model is available, False if it does not exist / cannot be pulled.
        Transport errors are propagated (they are infrastructure problems, not model problems).
        """
        if self._is_model_present(model):
            return True
        return self._pull_model(model)

    def get_server_version(self):
        """Returns the Ollama server version string, or None if it cannot be determined."""
        try:
            with urllib.request.urlopen(f"{self.host}/api/version", timeout=10) as resp:
                return json.loads(resp.read().decode()).get("version")
        except Exception:
            return None

    def _is_model_present(self, model):
        try:
            self.client.show(model)
            return True
        except ResponseError:
            print(f"Model {model} is not present on the server yet.")
            return False

    def _pull_model(self, model):
        """Pulls the model, streaming progress so that long downloads do not hit the request timeout."""
        try:
            print(f"Pulling model {model}...")
            last_reported = -10
            for progress in self.client.pull(model, stream=True):
                total = progress.get("total") or 0
                completed = progress.get("completed") or 0
                if total:
                    percent = int(completed * 100 / total)
                    if percent >= last_reported + 10:
                        print(f"  {model}: {progress.get('status', '')} {percent}%")
                        last_reported = percent
            print(f"Finished pulling model {model}.")
            return True
        except ResponseError as e:
            print(f"Failed to pull model {model}: {e}. The model may not exist or the Ollama "
                  f"server may be too old for it (server version: {self.get_server_version()}).")
            return False

    @staticmethod
    def build_options():
        """Builds the generation options sent with every request (also recorded in progress.json)."""
        options = {}
        if config.MAX_OUTPUT_TOKENS is not None:
            options["num_predict"] = config.MAX_OUTPUT_TOKENS
        if config.NUM_THREAD is not None:
            options["num_thread"] = config.NUM_THREAD
        if config.CONTEXT_WINDOW is not None:
            options["num_ctx"] = config.CONTEXT_WINDOW
        if config.SEED is not None:
            options["seed"] = config.SEED
        if config.TEMPERATURE is not None:
            options["temperature"] = config.TEMPERATURE
        return options
