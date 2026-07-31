"""Configuration for the local LLM endpoint.

Everything is read from the environment. In particular the model name is never
hardcoded (spec section 4.1): a missing ``LLM_MODEL`` is an error rather than a
silent fallback, because quietly picking a different model than the operator
intended would make results unreproducible.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

StructuredOutputMode = Literal["auto", "json_schema", "json_object"]
"""How to ask the server for JSON.

``json_schema`` constrains generation to a supplied JSON Schema, which is what
makes the catalog-id enum binding real rather than advisory. ``json_object``
only guarantees syntactically valid JSON. ``auto`` tries the first and falls
back to the second when the server rejects it.
"""

DEFAULT_BASE_URL = "http://localhost:11434/v1"
"""Ollama's OpenAI-compatible endpoint, the most common local default."""

STRUCTURED_OUTPUT_MODES: tuple[StructuredOutputMode, ...] = (
    "auto",
    "json_schema",
    "json_object",
)

ENV_BASE_URL = "LLM_BASE_URL"
ENV_MODEL = "LLM_MODEL"
ENV_API_KEY = "LLM_API_KEY"
ENV_TIMEOUT = "LLM_TIMEOUT"
ENV_TEMPERATURE = "LLM_TEMPERATURE"
ENV_MAX_TOKENS = "LLM_MAX_TOKENS"
ENV_STRUCTURED_OUTPUT = "LLM_STRUCTURED_OUTPUT"


class LLMConfigError(Exception):
    """Raised when the environment does not describe a usable endpoint."""


@dataclass(frozen=True)
class LLMConfig:
    """Where the local model lives and how to call it."""

    model: str
    base_url: str = DEFAULT_BASE_URL
    api_key: str = "local"
    """Sent as a bearer token. Local servers usually ignore it, but some proxies
    in front of them do not, so it is always sent."""

    timeout: float = 120.0
    """Seconds. Local inference on CPU is slow; the default is deliberately generous."""

    temperature: float = 0.0
    """Zero by default: the same request should produce the same assembly."""

    max_tokens: int = 2048
    """An ``AssemblySpec`` is a few dozen lines, so this is ample."""

    structured_output: StructuredOutputMode = "auto"

    @property
    def chat_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/chat/completions"


def load_config(
    *,
    model: str | None = None,
    base_url: str | None = None,
    env: dict[str, str] | None = None,
) -> LLMConfig:
    """Build a config from the environment, with optional explicit overrides.

    ``model`` and ``base_url`` come from CLI flags when given; everything else
    is environment-only.
    """
    source = os.environ if env is None else env

    resolved_model = model or source.get(ENV_MODEL, "").strip()
    if not resolved_model:
        msg = (
            f"no model configured: set {ENV_MODEL} (or pass --model). "
            "The model name is never hardcoded, so there is no default to fall "
            "back to. Example: LLM_MODEL=qwen2.5:7b"
        )
        raise LLMConfigError(msg)

    mode = source.get(ENV_STRUCTURED_OUTPUT, "auto").strip().lower()
    if mode not in STRUCTURED_OUTPUT_MODES:
        allowed = ", ".join(STRUCTURED_OUTPUT_MODES)
        msg = f"{ENV_STRUCTURED_OUTPUT}={mode!r} is not one of: {allowed}"
        raise LLMConfigError(msg)

    return LLMConfig(
        model=resolved_model,
        base_url=(base_url or source.get(ENV_BASE_URL) or DEFAULT_BASE_URL).strip(),
        api_key=source.get(ENV_API_KEY, "local"),
        timeout=_number(source, ENV_TIMEOUT, 120.0),
        temperature=_number(source, ENV_TEMPERATURE, 0.0),
        max_tokens=int(_number(source, ENV_MAX_TOKENS, 2048.0)),
        structured_output=mode,
    )


def _number(source: dict[str, str] | os._Environ[str], name: str, default: float) -> float:
    raw = source.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError as exc:
        msg = f"{name}={raw!r} is not a number"
        raise LLMConfigError(msg) from exc
