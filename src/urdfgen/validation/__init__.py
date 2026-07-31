"""Deterministic validation of an :class:`AssemblySpec`.

No check in this package uses an LLM, and none of them raise on an invalid
assembly: every problem comes back as a :class:`ValidationError` so the caller
-- the CLI, the repair loop, or the UI -- can decide what to do (spec sections
5 and 12).

The checks run in stages, because the later ones need what the earlier ones
guarantee. Spec-level checks come first; expansion-dependent checks only run
once the spec is structurally sound enough to expand at all.
"""

from __future__ import annotations

from urdfgen.builder.expand import ExpansionError, expand
from urdfgen.catalog.loader import Catalog
from urdfgen.ir.assembly import AssemblySpec
from urdfgen.validation import catalog_check, connection, kinematics, physics
from urdfgen.validation.result import (
    ValidationError,
    error,
    errors_only,
    has_errors,
    summarise,
    warning,
    warnings_only,
)

__all__ = [
    "ValidationError",
    "error",
    "errors_only",
    "has_errors",
    "summarise",
    "validate",
    "validate_spec_only",
    "warning",
    "warnings_only",
]


def validate_spec_only(spec: AssemblySpec, catalog: Catalog) -> list[ValidationError]:
    """Run the checks that need nothing but the spec and the catalog."""
    findings: list[ValidationError] = []
    for check in (*catalog_check.CHECKS, *connection.CHECKS, *kinematics.SPEC_CHECKS):
        findings.extend(check(spec, catalog))
    return findings


def validate(spec: AssemblySpec, catalog: Catalog) -> list[ValidationError]:
    """Run every check, returning findings in the order they were produced.

    Expansion-dependent checks are skipped when the spec-level checks already
    found errors: expanding a spec with an unknown module or a cycle would
    either fail outright or report cascading nonsense that buries the real
    problem.
    """
    findings = validate_spec_only(spec, catalog)
    if has_errors(findings):
        return findings

    try:
        robot = expand(spec, catalog)
    except ExpansionError as exc:
        findings.append(error("expansion_failed", str(exc), ""))
        return findings

    for robot_check in kinematics.ROBOT_CHECKS:
        findings.extend(robot_check(robot))
    for physics_check in physics.CHECKS:
        findings.extend(physics_check(robot))
    findings.extend(kinematics.check_roundtrip(robot, catalog))
    return findings
