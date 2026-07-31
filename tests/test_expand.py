"""Expansion engine tests (phase 1)."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from conftest import EXPECTED, FIXED_TIMESTAMP, load_spec
from urdfgen.builder.expand import (
    ExpandedJoint,
    ExpandedRobot,
    ExpansionError,
    apply_mirror,
    apply_parameters,
    expand,
    expand_module,
    global_name,
)
from urdfgen.builder.expression import ExpressionError, evaluate
from urdfgen.builder.urdf import build_urdf_string, format_float, mesh_uri
from urdfgen.catalog.loader import Catalog
from urdfgen.ir.assembly import AssemblySpec, Connection, ModuleInstance, PortRef, Pose

HALF_PI = math.pi / 2.0


def _spec(*connections: Connection, **kwargs: object) -> AssemblySpec:
    """A two-instance probe assembly: ``base`` with ``arm`` attached."""
    modules = [
        ModuleInstance(id="base", module="probe_base_v1"),
        ModuleInstance(id="arm", module="probe_arm_v1", **kwargs),
    ]
    return AssemblySpec(name="probe", root="base", modules=modules, connections=list(connections))


def _connect(parent_port: str, child_port: str = "plug", **kwargs: object) -> Connection:
    return Connection(
        parent=PortRef(instance="base", port=parent_port),
        child=PortRef(instance="arm", port=child_port),
        **kwargs,
    )


# --- namespacing ----------------------------------------------------------


def test_namespacing_prefixes_every_name(test_catalog: Catalog) -> None:
    robot = expand(_spec(_connect("socket_a")), test_catalog)
    assert {link.name for link in robot.links} == {
        "base_cube",
        "arm_stem",
        "arm_tip",
    }
    assert "arm_hinge" in {joint.name for joint in robot.joints}


def test_two_instances_of_one_module_do_not_collide(test_catalog: Catalog) -> None:
    spec = AssemblySpec(
        name="twin",
        root="base",
        modules=[
            ModuleInstance(id="base", module="probe_base_v1"),
            ModuleInstance(id="arm_a", module="probe_arm_v1"),
            ModuleInstance(id="arm_b", module="probe_arm_v1"),
        ],
        connections=[
            Connection(
                parent=PortRef(instance="base", port="socket_a"),
                child=PortRef(instance="arm_a", port="plug"),
            ),
            Connection(
                parent=PortRef(instance="base", port="socket_turned"),
                child=PortRef(instance="arm_b", port="plug"),
            ),
        ],
    )
    robot = expand(spec, test_catalog)
    assert len({link.name for link in robot.links}) == len(robot.links) == 5
    assert len({joint.name for joint in robot.joints}) == len(robot.joints) == 4


def test_name_map_supports_reverse_lookup(test_catalog: Catalog) -> None:
    robot = expand(_spec(_connect("socket_a")), test_catalog)
    assert robot.name_map[("arm", "hinge")] == "arm_hinge"
    assert robot.name_map[("base", "cube")] == "base_cube"
    assert global_name("arm", "tip") == "arm_tip"


def test_duplicate_instance_id_is_refused(test_catalog: Catalog) -> None:
    spec = AssemblySpec(
        name="dup",
        root="base",
        modules=[
            ModuleInstance(id="base", module="probe_base_v1"),
            ModuleInstance(id="base", module="probe_arm_v1"),
        ],
    )
    with pytest.raises(ExpansionError, match="duplicate module instance id"):
        expand(spec, test_catalog)


# --- connection transforms ------------------------------------------------


def test_connection_origin_is_parent_port_times_inverse_child_port(
    test_catalog: Catalog,
) -> None:
    """socket_a sits at x=+2; the arm's plug sits at z=-1 on its root link.

    ``T = T_parent . T_offset . T_child^-1`` therefore places the arm's stem at
    (2, 0, 1): two metres out along X, then one metre up to undo the plug
    offset.
    """
    robot = expand(_spec(_connect("socket_a")), test_catalog)
    joint = _joint(robot, "base_socket_a_to_arm")
    np.testing.assert_allclose(joint.origin.xyz, (2.0, 0.0, 1.0), atol=1e-12)
    np.testing.assert_allclose(joint.origin.rpy, (0.0, 0.0, 0.0), atol=1e-12)
    assert joint.parent == "base_cube"
    assert joint.child == "arm_stem"


def test_connection_origin_composes_rotation(test_catalog: Catalog) -> None:
    """socket_turned is yawed 90 degrees, so the plug offset rotates with it.

    The port frame is at (0, 0, 2) yawed by +pi/2. Undoing the plug's (0, 0, -1)
    offset in that rotated frame still moves along Z, so the stem lands at
    (0, 0, 3) with the yaw preserved.
    """
    robot = expand(_spec(_connect("socket_turned")), test_catalog)
    joint = _joint(robot, "base_socket_turned_to_arm")
    np.testing.assert_allclose(joint.origin.xyz, (0.0, 0.0, 3.0), atol=1e-12)
    np.testing.assert_allclose(joint.origin.rpy, (0.0, 0.0, HALF_PI), atol=1e-12)


def test_connection_offset_is_applied(test_catalog: Catalog) -> None:
    connection = _connect("socket_a", offset=Pose(xyz=(0.0, 0.5, 0.0)))
    robot = expand(_spec(connection), test_catalog)
    joint = _joint(robot, "base_socket_a_to_arm")
    np.testing.assert_allclose(joint.origin.xyz, (2.0, 0.5, 1.0), atol=1e-12)


def test_offset_rotation_is_composed_not_added(test_catalog: Catalog) -> None:
    """A rotated offset on a rotated port must compose as matrices.

    Adding the RPY triples would give the same yaw here but the wrong
    translation, which is exactly the failure mode spec 6.2 forbids.
    """
    connection = _connect(
        "socket_turned", offset=Pose(xyz=(1.0, 0.0, 0.0), rpy=(0.0, 0.0, HALF_PI))
    )
    robot = expand(_spec(connection), test_catalog)
    joint = _joint(robot, "base_socket_turned_to_arm")
    # The port yaws +90 deg, so the offset's local +X points along world +Y.
    np.testing.assert_allclose(joint.origin.xyz, (0.0, 1.0, 3.0), atol=1e-12)
    np.testing.assert_allclose(joint.origin.rpy, (0.0, 0.0, math.pi), atol=1e-12)


def test_connection_defaults_to_a_fixed_joint(test_catalog: Catalog) -> None:
    robot = expand(_spec(_connect("socket_a")), test_catalog)
    assert _joint(robot, "base_socket_a_to_arm").type == "fixed"


def test_connection_joint_type_can_be_requested(test_catalog: Catalog) -> None:
    robot = expand(_spec(_connect("socket_a", joint="continuous")), test_catalog)
    assert _joint(robot, "base_socket_a_to_arm").type == "continuous"


def test_connection_rejects_joint_types_that_need_limits(test_catalog: Catalog) -> None:
    with pytest.raises(ExpansionError, match="not usable for a connection"):
        expand(_spec(_connect("socket_a", joint="revolute")), test_catalog)


def test_child_must_attach_through_its_root_link(test_catalog: Catalog) -> None:
    with pytest.raises(ExpansionError, match="only the module's root_link"):
        expand(_spec(_connect("socket_a", child_port="far_plug")), test_catalog)


def test_unknown_port_is_reported(test_catalog: Catalog) -> None:
    spec = _spec(
        Connection(
            parent=PortRef(instance="base", port="nonexistent"),
            child=PortRef(instance="arm", port="plug"),
        )
    )
    with pytest.raises(ExpansionError, match="has no port 'nonexistent'"):
        expand(spec, test_catalog)


# --- ports and inertia ----------------------------------------------------


def test_unconnected_ports_are_tracked(test_catalog: Catalog) -> None:
    robot = expand(_spec(_connect("socket_a")), test_catalog)
    unconnected = {port.ref for port in robot.unconnected_ports()}
    assert "base.socket_a" not in unconnected
    assert "arm.plug" not in unconnected
    assert {"base.socket_b", "base.socket_turned", "arm.far_plug"} <= unconnected


def test_declared_inertia_is_never_recomputed(catalog: Catalog) -> None:
    """Spec 12: catalog inertia must survive expansion untouched."""
    robot = expand_module("leg_3dof_left_v1", catalog)
    thigh = robot.link("leg_3dof_left_v1_thigh")
    assert thigh is not None and thigh.inertial is not None
    assert thigh.inertial.inertia is not None
    assert thigh.inertial.inertia.ixx == pytest.approx(0.00118482)


def test_omitted_tensor_is_computed_from_the_primitive(test_catalog: Catalog) -> None:
    """probe_arm_v1's tip declares a mass but no tensor: appendix A fills it in."""
    robot = expand_module("probe_arm_v1", test_catalog)
    tip = robot.link("probe_arm_v1_tip")
    assert tip is not None and tip.inertial is not None
    assert tip.inertial.inertia is not None
    # Unit cube, mass 12: I = m(a^2 + b^2)/12 = 2 on every axis.
    assert tip.inertial.inertia.ixx == pytest.approx(2.0)
    assert tip.inertial.inertia.izz == pytest.approx(2.0)


def test_total_mass_sums_the_instances(test_catalog: Catalog) -> None:
    robot = expand(_spec(_connect("socket_a")), test_catalog)
    assert robot.total_mass == pytest.approx(6.0 + 2.0 + 12.0)


# --- parameters -----------------------------------------------------------


def test_parameters_drive_every_declared_target(test_catalog: Catalog) -> None:
    module = apply_parameters(test_catalog.get("probe_param_v1"), {"length": 1.6})
    rod = module.link("rod")
    tip = module.port("tip")
    assert rod is not None and tip is not None and rod.visual is not None
    assert rod.visual.geometry.length == pytest.approx(1.6)  # type: ignore[union-attr]
    assert rod.visual.origin.xyz[2] == pytest.approx(0.8)
    assert rod.inertial is not None
    assert rod.inertial.mass == pytest.approx(1.6)
    assert tip.origin.xyz[2] == pytest.approx(1.6)


def test_defaults_are_applied_when_no_value_is_given(test_catalog: Catalog) -> None:
    """Applying defaults keeps the authored YAML and the expressions in step."""
    module = apply_parameters(test_catalog.get("probe_param_v1"), {})
    tip = module.port("tip")
    assert tip is not None
    assert tip.origin.xyz[2] == pytest.approx(1.0)


def test_parameter_outside_its_range_is_refused(test_catalog: Catalog) -> None:
    with pytest.raises(ExpansionError, match="outside"):
        apply_parameters(test_catalog.get("probe_param_v1"), {"length": 5.0})


def test_unknown_parameter_is_refused(test_catalog: Catalog) -> None:
    with pytest.raises(ExpansionError, match="unknown parameters"):
        apply_parameters(test_catalog.get("probe_base_v1"), {"length": 1.0})


@pytest.mark.parametrize(
    ("expr", "expected"),
    [("value", 2.0), ("-value", -2.0), ("0.18 * value / 0.21", 0.18 * 2.0 / 0.21)],
)
def test_expression_arithmetic(expr: str, expected: float) -> None:
    assert evaluate(expr, {"value": 2.0}) == pytest.approx(expected)


@pytest.mark.parametrize(
    "expr",
    ["__import__('os').system('true')", "open('/etc/passwd')", "value.__class__", "[1, 2]"],
)
def test_expressions_reject_anything_but_arithmetic(expr: str) -> None:
    """Expressions come from editable YAML, so ``eval`` is never used."""
    with pytest.raises(ExpressionError):
        evaluate(expr, {"value": 1.0})


def test_expression_rejects_unknown_variables() -> None:
    with pytest.raises(ExpressionError, match="unknown variable"):
        evaluate("other * 2", {"value": 1.0})


# --- mirroring ------------------------------------------------------------


def test_mirroring_reproduces_the_dedicated_right_module(catalog: Catalog) -> None:
    """The right leg is authored as the exact Y mirror of the left one.

    Deriving it two ways and comparing is what proves the section 5.4 rules
    are implemented consistently with the hand-written catalog.
    """
    mirrored = apply_mirror(catalog.get("leg_3dof_left_v1"), "y")
    authored = catalog.get("leg_3dof_right_v1")

    for left, right in zip(mirrored.joints, authored.joints, strict=True):
        assert left.name == right.name
        np.testing.assert_allclose(left.origin.xyz, right.origin.xyz, atol=1e-12)
        np.testing.assert_allclose(left.axis, right.axis, atol=1e-12)
        assert left.limit is not None and right.limit is not None
        assert left.limit.lower == pytest.approx(right.limit.lower)
        assert left.limit.upper == pytest.approx(right.limit.upper)

    for left_link, right_link in zip(mirrored.links, authored.links, strict=True):
        assert left_link.inertial is not None and right_link.inertial is not None
        np.testing.assert_allclose(
            left_link.inertial.origin.xyz, right_link.inertial.origin.xyz, atol=1e-12
        )


def test_mirror_none_leaves_the_module_untouched(catalog: Catalog) -> None:
    module = catalog.get("leg_3dof_left_v1")
    assert apply_mirror(module, "none") is module


def test_mirrored_instance_expands(catalog: Catalog) -> None:
    spec = AssemblySpec(
        name="mirrored",
        root="body",
        modules=[
            ModuleInstance(id="body", module="body_quad_v1"),
            ModuleInstance(id="leg_fr", module="leg_3dof_left_v1", mirror="y"),
        ],
        connections=[
            Connection(
                parent=PortRef(instance="body", port="leg_mount_fr"),
                child=PortRef(instance="leg_fr", port="base"),
            )
        ],
    )
    robot = expand(spec, catalog)
    knee = _joint(robot, "leg_fr_knee_pitch")
    np.testing.assert_allclose(knee.axis, (0.0, -1.0, 0.0), atol=1e-12)
    assert knee.limit is not None
    assert (knee.limit.lower, knee.limit.upper) == pytest.approx((0.0, 2.36))


# --- URDF output ----------------------------------------------------------


def test_float_formatting_normalises_negative_zero() -> None:
    assert format_float(-0.0) == "0"
    assert format_float(0.0) == "0"
    assert format_float(1.0 / 3.0) == "0.333333"


def test_mesh_paths_become_package_uris() -> None:
    assert (
        mesh_uri("leg_3dof_left_v1", "meshes/shank.stl", "my_pkg")
        == "package://my_pkg/meshes/leg_3dof_left_v1/shank.stl"
    )


def test_materials_are_emitted_once(catalog: Catalog) -> None:
    """Four legs share the same materials; each must appear a single time."""
    xml = build_urdf_string(expand(load_spec("quadruped_a"), catalog), catalog)
    assert xml.count('<material name="aluminum">') == 1
    assert xml.count('<color rgba="0.75 0.76 0.78 1" />') == 1


def test_provenance_is_recorded(catalog: Catalog) -> None:
    xml = build_urdf_string(
        expand_module("body_quad_v1", catalog),
        catalog,
        source_text="a quadruped robot",
        timestamp=FIXED_TIMESTAMP,
    )
    assert "modules: body_quad_v1@1.0.0" in xml
    assert "generated: 2026-01-01T00:00:00+00:00" in xml
    assert "input: a quadruped robot" in xml


def test_comment_text_cannot_break_out_of_the_comment(catalog: Catalog) -> None:
    """Input text is user-supplied, so ``--`` must not terminate the comment early."""
    xml = build_urdf_string(
        expand_module("body_quad_v1", catalog),
        catalog,
        source_text="a -- b --> c",
        timestamp=FIXED_TIMESTAMP,
    )
    assert "input: a - b -> c" in xml
    assert xml.count("<!--") == xml.count("-->")
    # The document still parses, which it would not if a comment ended early.
    ET.fromstring(xml)


def test_quadruped_matches_the_golden_urdf(catalog: Catalog) -> None:
    """Byte-for-byte comparison against a reviewed reference output."""
    xml = build_urdf_string(
        expand(load_spec("quadruped_a"), catalog),
        catalog,
        source_text="quadruped_a",
        timestamp=FIXED_TIMESTAMP,
    )
    assert xml == (EXPECTED / "quadruped_a.urdf").read_text(encoding="utf-8")


def test_single_module_matches_the_golden_urdf(catalog: Catalog) -> None:
    xml = build_urdf_string(
        expand_module("leg_3dof_left_v1", catalog),
        catalog,
        source_text="leg_3dof_left_v1",
        timestamp=FIXED_TIMESTAMP,
    )
    assert xml == (EXPECTED / "leg_3dof_left_v1.urdf").read_text(encoding="utf-8")


# --- helpers --------------------------------------------------------------


def _joint(robot: ExpandedRobot, name: str) -> ExpandedJoint:
    joint = next((j for j in robot.joints if j.name == name), None)
    assert joint is not None, f"no joint named {name!r}"
    return joint
