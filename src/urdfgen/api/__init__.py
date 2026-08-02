"""HTTP API for the web app (spec section 7).

Importing this package needs FastAPI, which is an optional extra::

    uv sync --extra api

Everything else in ``urdfgen`` works without it, so the import lives behind the
``serve`` command rather than at the top of the CLI.
"""

from __future__ import annotations

from urdfgen.api.app import API_PREFIX, create_app, default_web_root

__all__ = ["API_PREFIX", "create_app", "default_web_root"]
