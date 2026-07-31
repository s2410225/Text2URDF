"""A stub OpenAI-compatible chat server, for testing the assembly layer.

Mocking ``LocalLLMClient`` would leave the part most likely to be wrong -- the
request body, and specifically whether the enum constraints really carry the
catalog -- untested. This serves real HTTP instead, records what it received,
and replies from a script.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


@dataclass
class Reply:
    """One scripted response."""

    content: str
    """The message content, exactly as a model would emit it."""

    status: int = 200
    """Non-200 exercises the client's error and fallback paths."""

    body: str | None = None
    """Overrides the whole response body, for malformed-envelope tests."""

    @classmethod
    def json_object(cls, payload: dict[str, Any], status: int = 200) -> Reply:
        return cls(content=json.dumps(payload, ensure_ascii=False), status=status)


@dataclass
class StubServer:
    """A running stub. Use :func:`stub_server` rather than constructing directly."""

    base_url: str
    replies: list[Reply]
    requests: list[dict[str, Any]] = field(default_factory=list)

    def request(self, index: int) -> dict[str, Any]:
        assert len(self.requests) > index, (
            f"expected at least {index + 1} request(s), got {len(self.requests)}"
        )
        return self.requests[index]

    def schema_of(self, index: int) -> dict[str, Any]:
        """The JSON Schema sent with request ``index``.

        Fails the test if that request did not carry one, which is the point:
        an unconstrained request is a bug, not a style choice.
        """
        response_format = self.request(index).get("response_format", {})
        assert response_format.get("type") == "json_schema", (
            f"request {index} did not use schema-constrained generation: {response_format}"
        )
        schema: dict[str, Any] = response_format["json_schema"]["schema"]
        return schema

    def response_format_types(self) -> list[str]:
        return [request.get("response_format", {}).get("type", "") for request in self.requests]


class _Handler(BaseHTTPRequestHandler):
    server: _Server

    def do_POST(self) -> None:
        """The name is fixed by ``BaseHTTPRequestHandler``'s dispatch."""
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        state = self.server.state
        state.requests.append(payload)

        index = len(state.requests) - 1
        reply = state.replies[index] if index < len(state.replies) else state.replies[-1]

        if reply.body is not None:
            body = reply.body.encode("utf-8")
        else:
            body = json.dumps(
                {
                    "id": "stub",
                    "object": "chat.completion",
                    "model": payload.get("model", ""),
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": reply.content},
                            "finish_reason": "stop",
                        }
                    ],
                }
            ).encode("utf-8")

        self.send_response(reply.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        """Silence the default stderr access log."""


class _Server(ThreadingHTTPServer):
    state: StubServer


SHUTDOWN_POLL_SECONDS = 0.01
"""``serve_forever`` defaults to polling twice a second, which would add half a
second to the teardown of every test that starts a stub."""


@contextmanager
def serving(replies: list[Reply]) -> Iterator[StubServer]:
    """Run a stub on an ephemeral port for the duration of the block."""
    httpd = _Server(("127.0.0.1", 0), _Handler)
    host, port = httpd.server_address[:2]
    state = StubServer(base_url=f"http://{host!s}:{port}/v1", replies=list(replies))
    httpd.state = state

    thread = threading.Thread(
        target=httpd.serve_forever, args=(SHUTDOWN_POLL_SECONDS,), daemon=True
    )
    thread.start()
    try:
        yield state
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)
