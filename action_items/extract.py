from __future__ import annotations

import json
from typing import Protocol

from .prompt import SYSTEM_PROMPT, build_user_prompt
from .schema import Output

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


class ChatClient(Protocol):
    """Anything that can turn (system prompt, user prompt) into a raw JSON string."""

    def complete(self, system_prompt: str, user_prompt: str) -> str: ...


class HFClient:
    """Chat client backed by the Hugging Face Inference API / a TGI endpoint."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        token: str | None = None,
        provider: str = "featherless-ai",
    ):
        from huggingface_hub import InferenceClient

        self._client = InferenceClient(model=model, token=token, provider=provider)

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        response = self._client.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            max_tokens=1024,
            temperature=0.0,
        )
        return response.choices[0].message.content


def extract(notes: str, attendees: list[str], client: ChatClient) -> Output:
    raw = client.complete(SYSTEM_PROMPT, build_user_prompt(notes, attendees))
    data = json.loads(raw)
    return Output.model_validate(data)
