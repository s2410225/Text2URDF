"""The HTTP boundary to a local OpenAI-compatible server.

This is the only module in the package that performs network access (spec
section 12). Everything it returns is untrusted: the caller validates it.

Local runtimes differ in how much of the OpenAI API they implement, so the
client is defensive in two specific ways. It falls back from schema-constrained
generation to plain JSON mode when the server rejects the former, and it
extracts JSON from responses that arrive wrapped in prose or code fences, which
smaller local models produce routinely.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import httpx

from urdfgen.llm.config import LLMConfig

Message = dict[str, str]
Schema = dict[str, Any]

_FALLBACK_STATUS_CODES = frozenset({400, 404, 422, 500, 501})
"""Statuses that can mean 'this server does not do json_schema'.

Local servers are inconsistent here: llama.cpp returns 400, some vLLM builds
500, older Ollama 404 on the parameter. Rather than guess from the message, any
of these triggers exactly one retry in plain JSON mode.
"""


class LLMError(Exception):
    """Raised when the local model could not be reached or did not answer usefully."""


@dataclass(frozen=True)
class Completion:
    """One model response, plus how it was actually obtained."""

    data: dict[str, Any]
    """The parsed JSON object."""

    used_schema: bool
    """False when the server refused schema-constrained generation.

    The caller cannot assume enum constraints were enforced in that case, which
    is why this is reported rather than hidden.
    """

    raw: str
    """The unparsed message content, kept for error messages."""


class LocalLLMClient:
    """A thin client for ``POST {base_url}/chat/completions``."""

    def __init__(self, config: LLMConfig, client: httpx.Client | None = None) -> None:
        self._config = config
        self._client = client or httpx.Client(timeout=config.timeout)
        self._owns_client = client is None
        self._schema_supported = config.structured_output != "json_object"

    def __enter__(self) -> LocalLLMClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    @property
    def config(self) -> LLMConfig:
        return self._config

    @property
    def schema_supported(self) -> bool:
        """Whether schema-constrained generation is still believed to work.

        Turns false permanently once a server refuses it, so callers can put
        the schema in the prompt from the start instead of discovering the
        refusal again on every call.
        """
        return self._schema_supported

    def complete_json(
        self,
        messages: list[Message],
        schema: Schema,
        schema_name: str,
        *,
        fallback_messages: list[Message] | None = None,
    ) -> Completion:
        """Ask for a JSON object matching ``schema`` and return it parsed.

        ``fallback_messages`` is used instead of ``messages`` if the server
        turns out not to support schema-constrained generation -- the prompt
        then has to carry the schema itself, and sending the unguided version
        first would waste a round trip on an answer of the wrong shape.

        Raises :class:`LLMError` for anything that prevents getting a JSON
        object back at all. A response that parses but is semantically wrong is
        not this layer's problem -- the validation layer handles that.
        """
        if self._schema_supported:
            response = self._post(self._payload(messages, schema, schema_name, use_schema=True))
            if response.status_code in _FALLBACK_STATUS_CODES and self._may_fall_back():
                self._schema_supported = False
            elif response.status_code >= 400:
                raise self._http_error(response)
            else:
                content = self._content(response)
                return Completion(data=self._parse(content), used_schema=True, raw=content)

        plain = fallback_messages if fallback_messages is not None else messages
        response = self._post(self._payload(plain, schema, schema_name, use_schema=False))
        if response.status_code >= 400:
            raise self._http_error(response)
        content = self._content(response)
        return Completion(data=self._parse(content), used_schema=False, raw=content)

    # --- request construction ---------------------------------------------

    def _may_fall_back(self) -> bool:
        """``json_schema`` mode is an explicit demand, so it never falls back."""
        return self._config.structured_output != "json_schema"

    def _payload(
        self, messages: list[Message], schema: Schema, schema_name: str, *, use_schema: bool
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._config.model,
            "messages": messages,
            "temperature": self._config.temperature,
            "max_tokens": self._config.max_tokens,
            "stream": False,
        }
        if use_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "schema": schema, "strict": True},
            }
        else:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def _post(self, payload: dict[str, Any]) -> httpx.Response:
        try:
            return self._client.post(
                self._config.chat_url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {self._config.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=self._config.timeout,
            )
        except httpx.ConnectError as exc:
            msg = (
                f"cannot reach a local LLM at {self._config.base_url}: {exc}. "
                "Start one first, for example 'ollama serve' (then "
                "'ollama pull <model>'), or point LLM_BASE_URL at your server."
            )
            raise LLMError(msg) from exc
        except httpx.TimeoutException as exc:
            msg = (
                f"the local LLM at {self._config.base_url} did not respond within "
                f"{self._config.timeout:g}s. Raise LLM_TIMEOUT, or use a smaller model."
            )
            raise LLMError(msg) from exc
        except httpx.HTTPError as exc:
            msg = f"request to {self._config.chat_url} failed: {exc}"
            raise LLMError(msg) from exc

    def _http_error(self, response: httpx.Response) -> LLMError:
        detail = response.text.strip()
        if len(detail) > 500:
            detail = detail[:500] + "..."
        return LLMError(
            f"the local LLM returned HTTP {response.status_code} for model "
            f"{self._config.model!r}: {detail or '(empty body)'}"
        )

    # --- response handling -------------------------------------------------

    def _content(self, response: httpx.Response) -> str:
        try:
            body = response.json()
        except ValueError as exc:
            msg = f"the local LLM returned a non-JSON body: {response.text[:200]!r}"
            raise LLMError(msg) from exc

        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            msg = (
                "the response did not look like an OpenAI chat completion "
                f"(no choices[0].message.content): {json.dumps(body)[:200]}"
            )
            raise LLMError(msg) from exc

        if not isinstance(content, str) or not content.strip():
            msg = "the local LLM returned an empty message"
            raise LLMError(msg)
        return content

    def _parse(self, content: str) -> dict[str, Any]:
        text = extract_json(content)
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            msg = f"the local LLM did not return valid JSON: {exc}. Body was: {content[:300]!r}"
            raise LLMError(msg) from exc
        if not isinstance(parsed, dict):
            msg = f"expected a JSON object, got {type(parsed).__name__}: {content[:200]!r}"
            raise LLMError(msg)
        return parsed


def extract_json(content: str) -> str:
    """Pull the JSON object out of a model message.

    Small local models frequently wrap their answer in ``` fences or precede it
    with a sentence of explanation even when asked not to. Rather than let that
    fail as a parse error, find the first balanced ``{...}`` and use it -- while
    respecting string literals, so a brace inside a description does not throw
    the scan off.
    """
    text = content.strip()

    fence = text.find("```")
    if fence != -1:
        after = text[fence + 3 :]
        newline = after.find("\n")
        if newline != -1:
            body = after[newline + 1 :]
            closing = body.find("```")
            text = (body if closing == -1 else body[:closing]).strip()

    start = text.find("{")
    if start == -1:
        return text

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return text[start:]
