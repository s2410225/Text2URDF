"""Tests that talk to a real local model.

Deselected by default (see ``addopts`` in ``pyproject.toml``). Run them against
your own server once one is up:

    ollama serve && ollama pull qwen2.5:7b
    LLM_BASE_URL=http://localhost:11434/v1 LLM_MODEL=qwen2.5:7b \\
        uv run pytest -m integration -q

These assert only what a correct assembly must satisfy, not a particular
answer: a small model has latitude in how it names instances, and pinning that
down would make the test about the model rather than the pipeline.
"""

from __future__ import annotations

import pytest

from urdfgen.builder.expand import expand
from urdfgen.catalog.loader import Catalog
from urdfgen.llm.client import LocalLLMClient
from urdfgen.llm.config import LLMConfigError, load_config
from urdfgen.llm.repair import assemble_with_repair
from urdfgen.validation import errors_only

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def client() -> LocalLLMClient:
    try:
        config = load_config()
    except LLMConfigError as exc:
        pytest.skip(str(exc))
    return LocalLLMClient(config)


@pytest.mark.parametrize(
    "request_text",
    ["4脚ロボットを作って", "a four-legged robot", "a quadruped with an arm on top"],
)
def test_a_real_model_produces_a_valid_assembly(
    client: LocalLLMClient, catalog: Catalog, request_text: str
) -> None:
    outcome = assemble_with_repair(request_text, catalog, client)

    assert outcome.succeeded, (
        f"{request_text!r} failed after {outcome.attempts} attempt(s): "
        + "; ".join(str(finding) for finding in errors_only(outcome.findings))
    )
    assert outcome.spec is not None

    robot = expand(outcome.spec, catalog)
    assert robot.links
    assert robot.movable_joints


def test_a_real_model_reports_a_missing_module(client: LocalLLMClient, catalog: Catalog) -> None:
    """Spec 4.4: no substituting when the catalog does not have the part."""
    outcome = assemble_with_repair("a four-legged robot with a lidar mast on top", catalog, client)

    assert outcome.spec is not None
    assert errors_only(outcome.findings) == []
    assert all("lidar" not in module.module for module in outcome.spec.modules)
