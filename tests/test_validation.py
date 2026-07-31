"""Validation layer tests (phase 2).

Every check gets at least one passing case and one failing case, per the phase
2 completion criterion.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from conftest import load_spec
from urdfgen.builder.expand import expand, expand_module
from urdfgen.catalog.loader import Catalog
from urdfgen.ir.assembly import AssemblySpec, Connection, ModuleInstance, PortRef
from urdfgen.validation import (
    ValidationError,
    errors_only,
    has_errors,
    summarise,
    validate,
    validate_spec_only,
    warnings_only,
)
from urdfgen.validation.catalog_check import (
    check_modules_exist,
    check_parameters,
    check_ports_exist,
)
from urdfgen.validation.connection import (
    check_child_uses_root_link,
    check_connection_joint_type,
    check_instance_ids_unique,
    check_interfaces_match,
    check_port_usage,
    check_root_exists,
    check_tree_structure,
)
from urdfgen.validation.kinematics import (
    check_handedness,
    check_mirroring,
    check_roundtrip,
    check_self_collision,
    check_unconnected_ports,
)
from urdfgen.validation.physics import (
    check_inertia_positive_definite,
    check_inertia_triangle_inequality,
    check_masses,
    check_total_mass,
)


def codes(findings: list[ValidationError]) -> set[str]:
    return {finding.code for finding in findings}


def probe(*connections: Connection, root: str = "base", **arm: object) -> AssemblySpec:
    """A ``probe_base_v1`` + ``probe_arm_v1`` assembly for the test catalog."""
    return AssemblySpec(
        name="probe",
        root=root,
        modules=[
            ModuleInstance(id="base", module="probe_base_v1"),
            ModuleInstance(id="arm", module="probe_arm_v1", **arm),
        ],
        connections=list(connections),
    )


def link(parent_port: str, child_port: str = "plug", **kwargs: object) -> Connection:
    return Connection(
        parent=PortRef(instance="base", port=parent_port),
        child=PortRef(instance="arm", port=child_port),
        **kwargs,
    )


# --- result helpers -------------------------------------------------------


def test_result_helpers_split_by_severity() -> None:
    from urdfgen.validation.result import error, warning

    findings = [error("a", "boom"), warning("b", "hmm")]
    assert has_errors(findings)
    assert codes(errors_only(findings)) == {"a"}
    assert codes(warnings_only(findings)) == {"b"}
    assert summarise(findings) == "1 error(s), 1 warning(s)"


def test_finding_renders_its_location() -> None:
    from urdfgen.validation.result import error

    assert str(error("x", "bad", "connections[0]")) == "[error] x at connections[0]: bad"


# --- 5.1 catalog consistency ----------------------------------------------


def test_known_modules_pass(test_catalog: Catalog) -> None:
    assert check_modules_exist(probe(), test_catalog) == []


def test_unknown_module_is_an_error(test_catalog: Catalog) -> None:
    spec = AssemblySpec(
        name="x", root="a", modules=[ModuleInstance(id="a", module="does_not_exist_v1")]
    )
    assert codes(check_modules_exist(spec, test_catalog)) == {"unknown_module"}


def test_unknown_version_is_an_error(test_catalog: Catalog) -> None:
    spec = AssemblySpec(
        name="x",
        root="a",
        modules=[ModuleInstance(id="a", module="probe_base_v1", version="9.9.9")],
    )
    assert codes(check_modules_exist(spec, test_catalog)) == {"unknown_version"}


def test_pinned_version_that_exists_passes(test_catalog: Catalog) -> None:
    spec = AssemblySpec(
        name="x",
        root="a",
        modules=[ModuleInstance(id="a", module="probe_base_v1", version="1.0.0")],
    )
    assert check_modules_exist(spec, test_catalog) == []


def test_known_ports_pass(test_catalog: Catalog) -> None:
    assert check_ports_exist(probe(link("socket_a")), test_catalog) == []


def test_unknown_port_is_an_error(test_catalog: Catalog) -> None:
    spec = probe(link("no_such_socket"))
    assert codes(check_ports_exist(spec, test_catalog)) == {"unknown_port"}


def test_connection_to_undeclared_instance_is_an_error(test_catalog: Catalog) -> None:
    spec = AssemblySpec(
        name="x",
        root="base",
        modules=[ModuleInstance(id="base", module="probe_base_v1")],
        connections=[
            Connection(
                parent=PortRef(instance="base", port="socket_a"),
                child=PortRef(instance="ghost", port="plug"),
            )
        ],
    )
    assert "unknown_instance" in codes(check_ports_exist(spec, test_catalog))


def test_parameter_within_range_passes(test_catalog: Catalog) -> None:
    spec = AssemblySpec(
        name="x",
        root="a",
        modules=[ModuleInstance(id="a", module="probe_param_v1", parameters={"length": 1.5})],
    )
    assert check_parameters(spec, test_catalog) == []


def test_parameter_out_of_range_is_an_error(test_catalog: Catalog) -> None:
    spec = AssemblySpec(
        name="x",
        root="a",
        modules=[ModuleInstance(id="a", module="probe_param_v1", parameters={"length": 9.0})],
    )
    assert codes(check_parameters(spec, test_catalog)) == {"parameter_out_of_range"}


def test_unknown_parameter_is_an_error(test_catalog: Catalog) -> None:
    spec = AssemblySpec(
        name="x",
        root="a",
        modules=[ModuleInstance(id="a", module="probe_base_v1", parameters={"nope": 1.0})],
    )
    assert codes(check_parameters(spec, test_catalog)) == {"unknown_parameter"}


# --- 5.2 connection consistency -------------------------------------------


def test_unique_instance_ids_pass(test_catalog: Catalog) -> None:
    assert check_instance_ids_unique(probe(), test_catalog) == []


def test_duplicate_instance_id_is_an_error(test_catalog: Catalog) -> None:
    spec = AssemblySpec(
        name="x",
        root="a",
        modules=[
            ModuleInstance(id="a", module="probe_base_v1"),
            ModuleInstance(id="a", module="probe_arm_v1"),
        ],
    )
    assert codes(check_instance_ids_unique(spec, test_catalog)) == {"duplicate_instance_id"}


def test_declared_root_passes(test_catalog: Catalog) -> None:
    assert check_root_exists(probe(), test_catalog) == []


def test_undeclared_root_is_an_error(test_catalog: Catalog) -> None:
    assert codes(check_root_exists(probe(root="ghost"), test_catalog)) == {"unknown_root"}


def test_each_port_used_once_passes(test_catalog: Catalog) -> None:
    assert check_port_usage(probe(link("socket_a")), test_catalog) == []


def test_port_used_twice_is_an_error(test_catalog: Catalog) -> None:
    spec = AssemblySpec(
        name="x",
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
                parent=PortRef(instance="base", port="socket_a"),
                child=PortRef(instance="arm_b", port="plug"),
            ),
        ],
    )
    assert codes(check_port_usage(spec, test_catalog)) == {"port_reused"}


def test_matching_interface_and_roles_pass(test_catalog: Catalog) -> None:
    assert check_interfaces_match(probe(link("socket_a")), test_catalog) == []


def test_interface_mismatch_is_an_error(test_catalog: Catalog) -> None:
    """socket_b is TEST_B, the arm's plug is TEST_A."""
    findings = check_interfaces_match(probe(link("socket_b")), test_catalog)
    assert codes(findings) == {"interface_mismatch"}


def test_same_role_on_both_ends_is_an_error(test_catalog: Catalog) -> None:
    """Both ends female: the spec's own example makes exactly this mistake."""
    spec = AssemblySpec(
        name="x",
        root="base",
        modules=[
            ModuleInstance(id="base", module="probe_base_v1"),
            ModuleInstance(id="other", module="probe_base_v1"),
        ],
        connections=[
            Connection(
                parent=PortRef(instance="base", port="socket_a"),
                child=PortRef(instance="other", port="socket_a"),
            )
        ],
    )
    assert "role_mismatch" in codes(check_interfaces_match(spec, test_catalog))


def test_child_attached_through_root_link_passes(test_catalog: Catalog) -> None:
    assert check_child_uses_root_link(probe(link("socket_a")), test_catalog) == []


def test_child_attached_through_other_link_is_an_error(test_catalog: Catalog) -> None:
    findings = check_child_uses_root_link(probe(link("socket_a", "far_plug")), test_catalog)
    assert codes(findings) == {"child_port_not_on_root_link"}


def test_tree_structure_passes(test_catalog: Catalog) -> None:
    assert check_tree_structure(probe(link("socket_a")), test_catalog) == []


def test_disconnected_instance_is_an_error(test_catalog: Catalog) -> None:
    assert codes(check_tree_structure(probe(), test_catalog)) == {"unreachable_instance"}


def test_two_parents_is_an_error(test_catalog: Catalog) -> None:
    spec = AssemblySpec(
        name="x",
        root="base",
        modules=[
            ModuleInstance(id="base", module="probe_base_v1"),
            ModuleInstance(id="mid", module="probe_base_v1"),
            ModuleInstance(id="arm", module="probe_arm_v1"),
        ],
        connections=[
            Connection(
                parent=PortRef(instance="base", port="socket_a"),
                child=PortRef(instance="mid", port="socket_a"),
            ),
            Connection(
                parent=PortRef(instance="base", port="socket_turned"),
                child=PortRef(instance="arm", port="plug"),
            ),
            Connection(
                parent=PortRef(instance="mid", port="socket_turned"),
                child=PortRef(instance="arm", port="plug_b"),
            ),
        ],
    )
    assert "multiple_parents" in codes(check_tree_structure(spec, test_catalog))


def test_cycle_is_an_error(test_catalog: Catalog) -> None:
    """``a -> b -> a`` leaves both unreachable from the root."""
    spec = AssemblySpec(
        name="x",
        root="base",
        modules=[
            ModuleInstance(id="base", module="probe_base_v1"),
            ModuleInstance(id="a", module="probe_arm_v1"),
            ModuleInstance(id="b", module="probe_arm_v1"),
        ],
        connections=[
            Connection(
                parent=PortRef(instance="a", port="far_plug"),
                child=PortRef(instance="b", port="plug"),
            ),
            Connection(
                parent=PortRef(instance="b", port="far_plug"),
                child=PortRef(instance="a", port="plug"),
            ),
        ],
    )
    assert "cycle_detected" in codes(check_tree_structure(spec, test_catalog))


def test_root_with_a_parent_is_an_error(test_catalog: Catalog) -> None:
    spec = AssemblySpec(
        name="x",
        root="arm",
        modules=[
            ModuleInstance(id="base", module="probe_base_v1"),
            ModuleInstance(id="arm", module="probe_arm_v1"),
        ],
        connections=[link("socket_a")],
    )
    assert "root_has_parent" in codes(check_tree_structure(spec, test_catalog))


def test_fixed_connection_joint_passes(test_catalog: Catalog) -> None:
    assert check_connection_joint_type(probe(link("socket_a")), test_catalog) == []
    assert check_connection_joint_type(probe(link("socket_a", joint="fixed")), test_catalog) == []


def test_revolute_connection_joint_is_an_error(test_catalog: Catalog) -> None:
    findings = check_connection_joint_type(probe(link("socket_a", joint="revolute")), test_catalog)
    assert codes(findings) == {"unsupported_connection_joint"}


# --- 5.3 physics ----------------------------------------------------------


def test_shipped_modules_have_valid_physics(catalog: Catalog) -> None:
    for module_id in catalog.ids():
        robot = expand_module(module_id, catalog)
        assert check_masses(robot) == []
        assert check_inertia_positive_definite(robot) == []
        assert check_inertia_triangle_inequality(robot) == []
        assert check_total_mass(robot) == []


def test_non_positive_definite_inertia_is_an_error(test_catalog: Catalog) -> None:
    robot = expand_module("probe_bad_inertia_v1", test_catalog)
    assert codes(check_inertia_positive_definite(robot)) == {"inertia_not_positive_definite"}


def test_triangle_inequality_violation_is_an_error(test_catalog: Catalog) -> None:
    robot = expand_module("probe_bad_inertia_v1", test_catalog)
    findings = check_inertia_triangle_inequality(robot)
    assert codes(findings) == {"inertia_triangle_inequality"}
    assert any("heavy" in finding.message for finding in findings)


def test_implausible_total_mass_is_a_warning(catalog: Catalog) -> None:
    """A one-tonne manipulator is structurally valid but physically absurd."""
    from urdfgen.validation.physics import TOTAL_MASS_RANGE

    robot = expand_module("arm_3dof_v1", catalog)
    heavy = replace(
        robot,
        links=tuple(
            replace(link, inertial=link.inertial.model_copy(update={"mass": TOTAL_MASS_RANGE[1]}))
            if link.inertial is not None
            else link
            for link in robot.links
        ),
    )
    findings = check_total_mass(heavy)
    assert codes(findings) == {"total_mass_out_of_range"}
    assert not findings[0].is_error


def test_link_without_mass_is_an_error(catalog: Catalog) -> None:
    robot = expand_module("body_quad_v1", catalog)
    stripped = replace(robot, links=tuple(replace(link, inertial=None) for link in robot.links))
    assert codes(check_masses(stripped)) == {"missing_mass"}


# --- 5.3 geometry, 5.4 mirroring ------------------------------------------


def test_no_free_ports_means_no_warning(catalog: Catalog) -> None:
    robot = expand(load_spec("quadruped_a"), catalog)
    fully_connected = replace(
        robot, ports=tuple(replace(port, connected=True) for port in robot.ports)
    )
    assert check_unconnected_ports(fully_connected) == []


def test_unconnected_ports_are_a_warning_not_an_error(catalog: Catalog) -> None:
    findings = check_unconnected_ports(expand(load_spec("quadruped_a"), catalog))
    assert codes(findings) == {"unconnected_port"}
    assert not has_errors(findings)


def test_no_mirroring_no_warning(catalog: Catalog) -> None:
    assert check_mirroring(load_spec("quadruped_a"), catalog) == []


def test_mirroring_always_warns(catalog: Catalog) -> None:
    spec = AssemblySpec(
        name="m",
        root="body",
        modules=[
            ModuleInstance(id="body", module="body_quad_v1"),
            ModuleInstance(id="leg_fr", module="leg_3dof_left_v1", mirror="y"),
        ],
    )
    findings = check_mirroring(spec, catalog)
    assert codes(findings) == {"mirrored_instance"}
    assert "left/right module is preferred" in findings[0].message


def test_correct_handedness_passes(catalog: Catalog) -> None:
    assert check_handedness(load_spec("quadruped_a"), catalog) == []


def test_left_module_on_a_right_mount_warns(catalog: Catalog) -> None:
    spec = AssemblySpec(
        name="swapped",
        root="body",
        modules=[
            ModuleInstance(id="body", module="body_quad_v1"),
            ModuleInstance(id="leg_fr", module="leg_3dof_left_v1"),
        ],
        connections=[
            Connection(
                parent=PortRef(instance="body", port="leg_mount_fr"),
                child=PortRef(instance="leg_fr", port="base"),
            )
        ],
    )
    findings = check_handedness(spec, catalog)
    assert codes(findings) == {"handedness_mismatch"}
    assert not has_errors(findings)


def test_quadruped_links_do_not_interfere(catalog: Catalog) -> None:
    assert check_self_collision(expand(load_spec("quadruped_a"), catalog)) == []


def test_overlapping_links_warn(test_catalog: Catalog) -> None:
    """Two unit cubes are placed at the same spot by a zero-length connection."""
    spec = AssemblySpec(
        name="overlap",
        root="base",
        modules=[
            ModuleInstance(id="base", module="probe_base_v1"),
            ModuleInstance(id="arm", module="probe_arm_v1"),
        ],
        connections=[
            Connection(
                parent=PortRef(instance="base", port="socket_a"),
                child=PortRef(instance="arm", port="plug"),
            )
        ],
    )
    robot = expand(spec, test_catalog)
    # Collapse every joint origin, stacking all three cubes on the same spot.
    # base_cube and arm_stem are adjacent so they are exempt, but arm_tip is
    # two joints away from base_cube and must be reported.
    collapsed = replace(
        robot,
        joints=tuple(
            replace(joint, origin=joint.origin.model_copy(update={"xyz": (0.0, 0.0, 0.0)}))
            for joint in robot.joints
        ),
    )
    findings = check_self_collision(collapsed)
    assert codes(findings) == {"link_interference"}
    assert any("arm_tip" in finding.message for finding in findings)
    assert not has_errors(findings)


# --- 5.5 round trip -------------------------------------------------------


def test_roundtrip_passes_for_the_shipped_catalog(catalog: Catalog) -> None:
    for module_id in catalog.ids():
        assert check_roundtrip(expand_module(module_id, catalog), catalog) == []


def test_roundtrip_passes_for_the_quadruped(catalog: Catalog) -> None:
    assert check_roundtrip(expand(load_spec("quadruped_a"), catalog), catalog) == []


def test_roundtrip_detects_a_builder_that_drops_content(
    catalog: Catalog, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stand in a wrong URDF for the builder's output.

    This check exists to catch a *builder* fault, so the fault has to be
    injected between expansion and parsing -- corrupting the expanded robot
    would corrupt both sides of the comparison and prove nothing.
    """
    from urdfgen.builder.urdf import build_urdf_string

    robot = expand(load_spec("quadruped_a"), catalog)
    wrong = build_urdf_string(expand_module("body_quad_v1", catalog), catalog)
    monkeypatch.setattr("urdfgen.validation.kinematics.build_urdf_string", lambda *a, **k: wrong)

    findings = check_roundtrip(robot, catalog)
    assert "roundtrip_link_count" in codes(findings)
    assert "roundtrip_joint_count" in codes(findings)
    assert "roundtrip_missing_joint" in codes(findings)
    assert has_errors(findings)


def test_roundtrip_detects_a_rewired_joint(
    catalog: Catalog, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same idea, but the counts still match and only the wiring differs."""
    from urdfgen.builder.urdf import build_urdf_string

    robot = expand(load_spec("quadruped_a"), catalog)
    rewired = list(robot.joints)
    rewired[0] = replace(rewired[0], parent=robot.root_link)
    wrong = build_urdf_string(replace(robot, joints=tuple(rewired)), catalog)
    monkeypatch.setattr("urdfgen.validation.kinematics.build_urdf_string", lambda *a, **k: wrong)

    findings = check_roundtrip(robot, catalog)
    assert codes(findings) == {"roundtrip_joint_mismatch"}
    assert has_errors(findings)


def test_roundtrip_detects_unparsable_output(
    catalog: Catalog, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("urdfgen.validation.kinematics.build_urdf_string", lambda *a, **k: "<robot")
    findings = check_roundtrip(expand_module("body_quad_v1", catalog), catalog)
    assert codes(findings) == {"urdf_parse_failed"}


# --- aggregation ----------------------------------------------------------


def test_valid_assembly_reports_no_errors(catalog: Catalog) -> None:
    findings = validate(load_spec("quadruped_a"), catalog)
    assert errors_only(findings) == []


def test_spec_level_errors_short_circuit_expansion(test_catalog: Catalog) -> None:
    """An unknown module must not cascade into expansion noise."""
    spec = AssemblySpec(
        name="x", root="a", modules=[ModuleInstance(id="a", module="does_not_exist_v1")]
    )
    assert codes(validate(spec, test_catalog)) == {"unknown_module"}


def test_validate_spec_only_skips_expansion_checks(catalog: Catalog) -> None:
    findings = validate_spec_only(load_spec("quadruped_a"), catalog)
    assert "unconnected_port" not in codes(findings)


def test_validation_never_raises_on_a_broken_spec(test_catalog: Catalog) -> None:
    """Spec 12: validation returns findings, it does not raise."""
    spec = AssemblySpec(
        name="broken",
        root="ghost",
        modules=[ModuleInstance(id="a", module="not_a_module_v1")],
        connections=[
            Connection(
                parent=PortRef(instance="ghost", port="nope"),
                child=PortRef(instance="a", port="nope"),
            )
        ],
    )
    findings = validate(spec, test_catalog)
    assert has_errors(findings)


@pytest.mark.parametrize("module_id", ["probe_bad_inertia_v1"])
def test_physics_errors_surface_through_validate(test_catalog: Catalog, module_id: str) -> None:
    spec = AssemblySpec(name="bad", root="a", modules=[ModuleInstance(id="a", module=module_id)])
    found = codes(validate(spec, test_catalog))
    assert "inertia_not_positive_definite" in found
    assert "inertia_triangle_inequality" in found
