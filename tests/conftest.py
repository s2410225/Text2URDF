"""Shared fixtures.

Tests run against two catalogs: the real one in ``modules/`` (so the shipped
catalog is exercised, not just a toy) and a tiny purpose-built one in
``tests/fixtures/modules`` for cases that need deliberately awkward geometry.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from urdfgen.catalog.loader import Catalog, load_catalog
from urdfgen.ir.assembly import AssemblySpec

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
SPECS = FIXTURES / "specs"
EXPECTED = FIXTURES / "expected"

FIXED_TIMESTAMP = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
"""Frozen clock, so golden URDF files can be compared byte for byte."""


@pytest.fixture(scope="session")
def catalog() -> Catalog:
    """The catalog shipped with the repository."""
    return load_catalog(REPO_ROOT / "modules")


@pytest.fixture(scope="session")
def test_catalog() -> Catalog:
    """The minimal catalog used for edge cases."""
    return load_catalog(FIXTURES / "modules")


def load_spec(name: str) -> AssemblySpec:
    """Read an ``AssemblySpec`` from ``tests/fixtures/specs``."""
    path = SPECS / name if name.endswith(".json") else SPECS / f"{name}.json"
    return AssemblySpec.model_validate(json.loads(path.read_text(encoding="utf-8")))
