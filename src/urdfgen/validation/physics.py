"""Mass and inertia checks (spec section 5.3).

These operate on the expanded robot rather than the spec, because that is where
mirroring and parameters have already been applied -- a tensor that is valid in
the catalog can still be wrong after transformation.
"""

from __future__ import annotations

import numpy as np

from urdfgen.builder.expand import ExpandedRobot
from urdfgen.builder.transform import inertia_to_matrix
from urdfgen.validation.result import ValidationError, error, warning

EIGENVALUE_TOLERANCE = 1e-12
"""Principal moments below this are treated as non-positive."""

TRIANGLE_TOLERANCE = 1e-9
"""Slack on the triangle inequality, to absorb rounding in authored values."""

CATEGORY_MASS_RANGE: dict[str, tuple[float, float]] = {
    "body": (0.05, 500.0),
    "leg": (0.02, 200.0),
    "arm": (0.02, 300.0),
    "gripper": (0.005, 50.0),
    "wheel": (0.005, 100.0),
    "sensor": (0.001, 20.0),
    "adapter": (0.001, 50.0),
}

TOTAL_MASS_RANGE: tuple[float, float] = (0.05, 1000.0)
"""Plausible total mass for a robot this system builds, in kilograms."""


def check_masses(robot: ExpandedRobot) -> list[ValidationError]:
    """Every link must declare a positive mass."""
    findings: list[ValidationError] = []
    for link in robot.links:
        if link.inertial is None:
            findings.append(
                error(
                    "missing_mass",
                    f"link {link.name!r} (module {link.module_id!r}) declares no inertial "
                    "block, so it has no mass",
                    f"links.{link.name}",
                )
            )
        elif link.inertial.mass <= 0.0:
            findings.append(
                error(
                    "non_positive_mass",
                    f"link {link.name!r} has mass {link.inertial.mass}; mass must be > 0",
                    f"links.{link.name}",
                )
            )
    return findings


def check_inertia_positive_definite(robot: ExpandedRobot) -> list[ValidationError]:
    """The inertia tensor must be symmetric positive definite."""
    findings: list[ValidationError] = []
    for link in robot.links:
        if link.inertial is None or link.inertial.inertia is None:
            continue
        eigenvalues = _principal_moments(robot, link.name)
        if eigenvalues is None:
            continue
        if float(np.min(eigenvalues)) <= EIGENVALUE_TOLERANCE:
            findings.append(
                error(
                    "inertia_not_positive_definite",
                    f"link {link.name!r} has principal moments "
                    f"{_format(eigenvalues)}; the inertia tensor must be positive definite",
                    f"links.{link.name}.inertial",
                )
            )
    return findings


def check_inertia_triangle_inequality(robot: ExpandedRobot) -> list[ValidationError]:
    """Principal moments must satisfy ``Ia + Ib >= Ic`` for every pairing.

    A tensor can be positive definite and still describe no physical body; this
    is the check that catches a mistyped moment.
    """
    findings: list[ValidationError] = []
    for link in robot.links:
        if link.inertial is None or link.inertial.inertia is None:
            continue
        eigenvalues = _principal_moments(robot, link.name)
        if eigenvalues is None:
            continue
        first, second, third = sorted(float(value) for value in eigenvalues)
        if first + second < third - TRIANGLE_TOLERANCE:
            findings.append(
                error(
                    "inertia_triangle_inequality",
                    f"link {link.name!r} has principal moments {_format(eigenvalues)}; "
                    f"{first:.6g} + {second:.6g} < {third:.6g} describes no physical body",
                    f"links.{link.name}.inertial",
                )
            )
    return findings


def check_total_mass(robot: ExpandedRobot) -> list[ValidationError]:
    """Warn when the assembled mass is implausible for a buildable robot."""
    total = robot.total_mass
    low, high = TOTAL_MASS_RANGE
    if low <= total <= high:
        return []
    return [
        warning(
            "total_mass_out_of_range",
            f"total mass {total:.3f} kg is outside the expected range "
            f"{low}-{high} kg; check the module definitions",
            "links",
        )
    ]


def _principal_moments(robot: ExpandedRobot, link_name: str) -> np.ndarray | None:
    link = robot.link(link_name)
    if link is None or link.inertial is None or link.inertial.inertia is None:
        return None
    tensor = inertia_to_matrix(link.inertial.inertia)
    return np.linalg.eigvalsh(tensor)


def _format(values: np.ndarray) -> str:
    return "[" + ", ".join(f"{float(value):.6g}" for value in values) + "]"


CHECKS = (
    check_masses,
    check_inertia_positive_definite,
    check_inertia_triangle_inequality,
    check_total_mass,
)
