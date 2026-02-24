"""OpenAI-compatible LLM client.

Supports any backend with an OpenAI-compatible Chat Completion endpoint:
  - OpenAI (api.openai.com)
  - DeepSeek (api.deepseek.com)
  - Local vLLM / Ollama (localhost:8000 / localhost:11434)
  - Azure OpenAI

Usage example::

    client = LLMClient(
        base_url="https://api.openai.com/v1",
        api_key=os.environ["OPENAI_API_KEY"],
        model="gpt-4o-mini",
    )
    result = client.chat_json([{"role": "user", "content": "Hello"}])
"""
from __future__ import annotations

import json
import time
from typing import Optional

try:
    from openai import OpenAI as _OpenAI

    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False


class LLMClient:
    """Thin wrapper around the OpenAI Python SDK for chat completions."""

    def __init__(
        self,
        base_url: str = "https://api.openai.com/v1",
        api_key: str = "dummy",
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        timeout: float = 30.0,
    ) -> None:
        if not _OPENAI_AVAILABLE:
            raise ImportError(
                "openai package is required. Install with: pip install openai"
            )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = _OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    def chat(
        self,
        messages: list[dict],
        response_format: Optional[dict] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Send messages and return assistant content string."""
        kwargs: dict = dict(
            model=self.model,
            messages=messages,
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=self.max_tokens,
        )
        if response_format is not None:
            kwargs["response_format"] = response_format
        response = self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    def chat_json(
        self,
        messages: list[dict],
        max_retries: int = 2,
        temperature: Optional[float] = None,
    ) -> dict:
        """Send messages, parse JSON response, with retry on parse failure."""
        for attempt in range(max_retries + 1):
            try:
                content = self.chat(
                    messages,
                    response_format={"type": "json_object"},
                    temperature=temperature,
                )
                return json.loads(content)
            except json.JSONDecodeError as exc:
                if attempt < max_retries:
                    print(f"[LLMClient] JSON parse error (attempt {attempt+1}): {exc}")
                    time.sleep(1.0)
                else:
                    print(f"[LLMClient] JSON parse failed after {max_retries+1} attempts.")
                    return {}
            except Exception as exc:
                if attempt < max_retries:
                    print(f"[LLMClient] API error (attempt {attempt+1}): {exc}")
                    time.sleep(2.0)
                else:
                    print(f"[LLMClient] API error: {exc}")
                    return {}
        return {}
