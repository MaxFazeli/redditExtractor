from dataclasses import dataclass
from typing import Optional, Any

@dataclass
class LlmResponse:
    ok: bool
    finish_reason: Optional[str] = None
    prompt_token_count: Optional[int] = None
    message: Optional[str] = None
    message_token_count: Optional[int] = None
    duration_seconds: Optional[float] = None
    error: Optional[str] = None
    llm_full_response: Optional[Any] = None
    llm_full_response_json: Optional[Any] = None
