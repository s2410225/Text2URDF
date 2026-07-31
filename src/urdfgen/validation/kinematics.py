"""Geometric and round-trip checks (spec sections 5.3 to 5.5).

Everything here needs the *expanded* robot: unconnected ports, link placement
and the generated XML only exist after expansion.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass

import numpy as np
import yourdfpy

from urdfgen.builder.expand import ExpandedRobot
from urdfgen.builder.transform import compose, pose_to_matrix
from urdfgen.builder.urdf import build_urdf_string
from urdfgen.catalog.loader import Catalog
from urdfgen.catalog.schema import (
    BoxGeometry,
    CylinderGeometry,
    Geometry,
    SphereGeometry,
)
from urdfgen.ir.assembly import AssemblySpec
from urdfgen.validation.result import ValidationError, error, warning

Matrix = np.ndarray

OVERLAP_TOLERANCE = 1e-6
"""Boxes touching within this distance are treated as adjacent, not overlapping."""

_LEFT_PORT = re.compile(r"(^|_)(fl|rl|l|left)$|left", re.IGNORECASE)
_RIGHT_PORT = re.compile(r"(^|_)(fr|rr|r|right)$|right", re.IGNORECASE)


def check_unconnected_ports(robot: ExpandedRobot) -> list[ValidationError]:
    """Report ports left free. This is often intentional, so it is a warning."""
    unconnected = robot.unconnected_ports()
    if not unconnected:
        return []
    listed = ", ".join(port.ref for port in unconnected)
    return [
        warning(
            "unconnected_port",
            f"{len(unconnected)} port(s) are left unconnected: {listed}",
            "connections",
        )
    ]


def check_mirroring(spec: AssemblySpec, catalog: Catalog) -> list[ValidationError]:
    """Always warn when an instance is mirrored (spec section 5.4)."""
    del catalog
    return [
        warning(
            "mirrored_instance",
            f"instance {instance.id!r} mirrors {instance.module!r} about {instance.mirror}; "
            "a dedicated left/right module is preferred, because negative mesh scales "
            "flip surface normals in some physics engines",
            f"modules[{index}].mirror",
        )
        for index, instance in enumerate(spec.modules)
        if instance.mirror != "none"
    ]


def check_handedness(spec: AssemblySpec, catalog: Catalog) -> list[ValidationError]:
    """Warn when a ``_left`` module is attached to a right-hand port, or vice versa.

    Purely a naming heuristic -- it cannot prove the geometry is wrong, which
    is exactly why it is a warning and not an error. Mirrored instances are
    exempt: flipping the module is a deliberate answer to this very problem,
    and :func:`check_mirroring` already reports it.
    """
    del catalog
    findings: list[ValidationError] = []
    for index, connection in enumerate(spec.connections):
        instance = spec.instance(connection.child.instance)
        if instance is None or instance.mirror != "none":
            continue
        module_side = _side_of_module(instance.module)
        port_side = _side_of_port(connection.parent.port)
        if module_side is None or port_side is None or module_side == port_side:
            continue
        findings.append(
            warning(
                "handedness_mismatch",
                f"{instance.module!r} is a {module_side}-hand module but is attached to "
                f"{connection.parent}, which is a {port_side}-hand mount; check that the "
                f"correct _{port_side} module was chosen",
                f"connections[{index}]",
            )
        )
    return findings


def check_self_collision(robot: ExpandedRobot) -> list[ValidationError]:
    """Coarse AABB overlap test with every joint at zero (spec section 5.3).

    Adjacent links are skipped: parts joined by a joint are supposed to touch.
    Mesh geometry is skipped too, since the catalog carries no mesh assets to
    measure.
    """
    poses = forward_kinematics(robot)
    boxes: dict[str, AABB] = {}
    for link in robot.links:
        if link.collision is None or link.name not in poses:
            continue
        placement = compose(poses[link.name], pose_to_matrix(link.collision.origin))
        box = _world_aabb(link.collision.geometry, placement)
        if box is not None:
            boxes[link.name] = box

    adjacent = {
        frozenset((joint.parent, joint.child))
        for joint in robot.joints
        if joint.parent != joint.child
    }

    findings: list[ValidationError] = []
    names = sorted(boxes)
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            if frozenset((first, second)) in adjacent:
                continue
            if _overlaps(boxes[first], boxes[second]):
                findings.append(
                    warning(
                        "link_interference",
                        f"links {first!r} and {second!r} have overlapping bounding boxes "
                        "with all joints at zero; this is a coarse test and may be a "
                        "false positive",
                        "links",
                    )
                )
    return findings


def check_roundtrip(robot: ExpandedRobot, catalog: Catalog) -> list[ValidationError]:
    """Re-parse the generated URDF and compare it to the expansion (spec 5.5).

    This is the check that catches a builder bug rather than a spec bug: if the
    XML we emit does not read back as the tree we expanded, nothing downstream
    can be trusted.
    """
    try:
        xml = build_urdf_string(robot, catalog)
    except (ValueError, KeyError) as exc:
        return [error("urdf_build_failed", f"could not generate URDF: {exc}", "")]

    try:
        parsed = yourdfpy.URDF.load(
            io.BytesIO(xml.encode("utf-8")),
            build_scene_graph=False,
            build_collision_scene_graph=False,
            load_meshes=False,
            load_collision_meshes=False,
        )
    except Exception as exc:
        # yourdfpy raises a variety of types depending on what is malformed;
        # any of them means the same thing here.
        return [error("urdf_parse_failed", f"generated URDF does not parse: {exc}", "")]

    findings: list[ValidationError] = []
    if len(parsed.link_map) != len(robot.links):
        findings.append(
            error(
                "roundtrip_link_count",
                f"expansion produced {len(robot.links)} links but the generated URDF "
                f"parses as {len(parsed.link_map)}",
                "",
            )
        )
    if len(parsed.joint_map) != len(robot.joints):
        findings.append(
            error(
                "roundtrip_joint_count",
                f"expansion produced {len(robot.joints)} joints but the generated URDF "
                f"parses as {len(parsed.joint_map)}",
                "",
            )
        )

    expected = {joint.name: (joint.parent, joint.child) for joint in robot.joints}
    actual = {name: (joint.parent, joint.child) for name, joint in parsed.joint_map.items()}
    for name, pair in sorted(expected.items()):
        if name not in actual:
            findings.append(
                error("roundtrip_missing_joint", f"joint {name!r} is missing from the URDF", "")
            )
        elif actual[name] != pair:
            findings.append(
                error(
                    "roundtrip_joint_mismatch",
                    f"joint {name!r} connects {actual[name]} in the URDF but "
                    f"{pair} after expansion",
                    "",
                )
            )
    return findings


# --- geometry helpers -----------------------------------------------------


@dataclass(frozen=True)
class AABB:
    """An axis-aligned bounding box in world coordinates."""

    low: Matrix
    high: Matrix


def forward_kinematics(robot: ExpandedRobot) -> dict[str, Matrix]:
    """World transform of every link with all joint values at zero.

    Links unreachable from the root are omitted rather than guessed at; the
    connection checks report the broken structure that caused it.
    """
    children: dict[str, list[tuple[str, Matrix]]] = {}
    for joint in robot.joints:
        children.setdefault(joint.parent, []).append((joint.child, pose_to_matrix(joint.origin)))

    poses: dict[str, Matrix] = {robot.root_link: np.eye(4)}
    stack = [robot.root_link]
    while stack:
        parent = stack.pop()
        for child, origin in children.get(parent, ()):
            if child in poses:
                continue  # a cycle; the connection checks report it
            poses[child] = compose(poses[parent], origin)
            stack.append(child)
    return poses


def _world_aabb(geometry: Geometry, transform: Matrix) -> AABB | None:
    """Bounding box of a primitive after ``transform``. ``None`` for meshes."""
    half = _half_extents(geometry)
    if half is None:
        return None

    corners = np.array(
        [
            [sx * half[0], sy * half[1], sz * half[2]]
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ]
    )
    rotated = corners @ transform[:3, :3].T + transform[:3, 3]
    return AABB(low=rotated.min(axis=0), high=rotated.max(axis=0))


def _half_extents(geometry: Geometry) -> tuple[float, float, float] | None:
    if isinstance(geometry, BoxGeometry):
        return (geometry.size[0] / 2.0, geometry.size[1] / 2.0, geometry.size[2] / 2.0)
    if isinstance(geometry, CylinderGeometry):
        return (geometry.radius, geometry.radius, geometry.length / 2.0)
    if isinstance(geometry, SphereGeometry):
        return (geometry.radius, geometry.radius, geometry.radius)
    return None


def _overlaps(first: AABB, second: AABB) -> bool:
    return bool(
        np.all(first.low < second.high - OVERLAP_TOLERANCE)
        and np.all(second.low < first.high - OVERLAP_TOLERANCE)
    )


def _side_of_module(module_id: str) -> str | None:
    if "_left" in module_id:
        return "left"
    if "_right" in module_id:
        return "right"
    return None


def _side_of_port(port_name: str) -> str | None:
    if _LEFT_PORT.search(port_name):
        return "left"
    if _RIGHT_PORT.search(port_name):
        return "right"
    return None


SPEC_CHECKS = (check_mirroring, check_handedness)
ROBOT_CHECKS = (check_unconnected_ports, check_self_collision)
