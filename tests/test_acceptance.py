"""Acceptance tests from spec section 11.

The table maps a natural-language request to the ``AssemblySpec`` the assembly
layer is expected to produce. Phase 3 will drive the same table through a
mocked LLM; until then each case supplies the spec directly, so what is being
asserted is the deterministic half: validation and URDF generation.

Cases 1 to 6 describe what a *correct* assembly of that request looks like.
Cases 7 and 8 are malformed specs a model might emit, and assert that the
validation layer rejects them and no URDF is produced.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import pytest
import yourdfpy

from conftest import SPECS, load_spec
from stub_llm import Reply, StubServer, serving
from urdfgen.builder.expand import expand
from urdfgen.builder.urdf import build_urdf_string
from urdfgen.catalog.loader import Catalog
from urdfgen.ir.assembly import AssemblySpec
from urdfgen.llm.client import LocalLLMClient
from urdfgen.llm.config import LLMConfig
from urdfgen.llm.repair import AssemblyOutcome, assemble_with_repair
from urdfgen.validation import ValidationError, errors_only, validate, warnings_only


@dataclass(frozen=True)
class Case:
    """One row of the acceptance table."""

    number: int
    request: str
    spec_name: str
    """Fixture under ``tests/fixtures/specs`` holding the expected assembly."""


CASES = [
    Case(1, "a four-legged robot", "quadruped_a"),
    Case(2, "a four-legged robot with an arm on top", "quadruped_with_arm"),
    Case(3, "put a gripper on the end of the arm", "quadruped_arm_gripper"),
    Case(4, "a two-legged robot", "biped"),
    Case(5, "a four-legged robot with a lidar mast", "missing_module"),
    Case(6, "a four-legged robot", "swapped_handedness"),
]

INVALID_CASES = [
    Case(7, "a leg bolted to the payload flange", "bad_interface"),
    Case(8, "two legs on the same mount", "double_connection"),
]


def codes(findings: list[ValidationError]) -> set[str]:
    return {finding.code for finding in findings}


@pytest.fixture(params=CASES, ids=lambda case: f"{case.number:02d}_{case.spec_name}")
def case(request: pytest.FixtureRequest) -> Case:
    assert isinstance(request.param, Case)
    return request.param


def test_valid_cases_produce_no_errors(case: Case, catalog: Catalog) -> None:
    spec = load_spec(case.spec_name)
    findings = validate(spec, catalog)
    assert errors_only(findings) == [], f"case {case.number} ({case.request})"


def test_valid_cases_build_a_parsable_urdf(case: Case, catalog: Catalog) -> None:
    spec = load_spec(case.spec_name)
    robot = expand(spec, catalog)
    xml = build_urdf_string(robot, catalog, source_text=case.request)

    assert xml.startswith('<?xml version="1.0"?>')
    parsed = yourdfpy.URDF.load(
        io.BytesIO(xml.encode("utf-8")),
        build_scene_graph=False,
        build_collision_scene_graph=False,
        load_meshes=False,
        load_collision_meshes=False,
    )
    assert len(parsed.link_map) == len(robot.links)
    assert len(parsed.joint_map) == len(robot.joints)


# --- case 1: four-legged robot --------------------------------------------


def test_case_1_quadruped_structure(catalog: Catalog) -> None:
    """One body, four legs, twelve movable joints, zero errors."""
    spec = load_spec("quadruped_a")
    robot = expand(spec, catalog)

    assert sum(1 for m in spec.modules if m.module == "body_quad_v1") == 1
    assert sum(1 for m in spec.modules if m.module.startswith("leg_3dof")) == 4
    assert len(robot.movable_joints) == 12
    assert len(robot.links) == 17
    assert errors_only(validate(spec, catalog)) == []


# --- case 2: arm on top ---------------------------------------------------


def test_case_2_arm_is_fixed_to_the_top_port(catalog: Catalog) -> None:
    spec = load_spec("quadruped_with_arm")
    robot = expand(spec, catalog)

    mount = next(joint for joint in robot.joints if joint.name.startswith("body_top_to_"))
    assert mount.type == "fixed"
    assert mount.parent == "body_chassis"
    assert mount.child == "arm_main_arm_base"
    assert len(robot.movable_joints) == 15  # 12 leg + 3 arm
    assert errors_only(validate(spec, catalog)) == []


# --- case 3: gripper on the arm -------------------------------------------


def test_case_3_gripper_hangs_off_the_arm_tool_port(catalog: Catalog) -> None:
    spec = load_spec("quadruped_arm_gripper")
    robot = expand(spec, catalog)

    mount = next(joint for joint in robot.joints if joint.name.startswith("arm_main_tool_to_"))
    assert mount.parent == "arm_main_forearm"
    assert mount.child == "gripper_palm"

    prismatic = [joint for joint in robot.joints if joint.type == "prismatic"]
    assert {joint.name for joint in prismatic} == {
        "gripper_finger_left_slide",
        "gripper_finger_right_slide",
    }
    assert len(robot.movable_joints) == 17  # 12 leg + 3 arm + 2 finger
    assert errors_only(validate(spec, catalog)) == []


# --- case 4: two-legged robot ---------------------------------------------


def test_case_4_biped_leaves_two_mounts_free(catalog: Catalog) -> None:
    spec = load_spec("biped")
    findings = validate(spec, catalog)

    assert errors_only(findings) == []
    unconnected = [f for f in warnings_only(findings) if f.code == "unconnected_port"]
    assert len(unconnected) == 1
    assert "body.leg_mount_rl" in unconnected[0].message
    assert "body.leg_mount_rr" in unconnected[0].message


# --- case 5: a module the catalog does not have ---------------------------


def test_case_5_missing_module_is_recorded_in_notes(catalog: Catalog) -> None:
    """The assembly layer must not substitute something else (spec 4.4).

    The partial robot still has to validate, so the user gets something they
    can edit rather than an outright failure.
    """
    spec = load_spec("missing_module")
    assert spec.notes is not None
    assert "lidar" in spec.notes.lower()
    assert not any("lidar" in module.module for module in spec.modules)
    assert errors_only(validate(spec, catalog)) == []


# --- case 6: left/right mix-up --------------------------------------------


def test_case_6_left_module_on_right_mount_warns(catalog: Catalog) -> None:
    spec = load_spec("swapped_handedness")
    findings = validate(spec, catalog)

    assert errors_only(findings) == []
    mismatches = [f for f in findings if f.code == "handedness_mismatch"]
    assert len(mismatches) == 2
    assert all("leg_3dof_left_v1" in f.message for f in mismatches)


def test_case_6_swapped_legs_also_trip_the_geometric_check(catalog: Catalog) -> None:
    """The naming heuristic and the AABB test agree, independently.

    A left leg on a right mount folds inwards, so its thigh ends up inside the
    chassis. Catching it twice is what makes the warning trustworthy.
    """
    interference = [
        finding
        for finding in validate(load_spec("swapped_handedness"), catalog)
        if finding.code == "link_interference"
    ]
    assert {"leg_fr_thigh", "leg_rr_thigh"} <= {
        word.strip("'") for finding in interference for word in finding.message.split()
    }


def test_case_6_mirroring_is_the_alternative(catalog: Catalog) -> None:
    """Mirroring fixes the geometry, so only the mirror warning remains."""
    findings = validate(load_spec("mirrored_legs"), catalog)

    assert errors_only(findings) == []
    assert len([f for f in findings if f.code == "mirrored_instance"]) == 2
    assert [f for f in findings if f.code == "handedness_mismatch"] == []
    assert [f for f in findings if f.code == "link_interference"] == []


# --- case 7: interface mismatch -------------------------------------------


def test_case_7_interface_mismatch_is_an_error(catalog: Catalog) -> None:
    spec = load_spec("bad_interface")
    findings = validate(spec, catalog)

    assert codes(errors_only(findings)) == {"interface_mismatch"}


def test_case_7_no_urdf_is_written(tmp_path: Path) -> None:
    """The CLI must refuse to build once validation reports an error."""
    from urdfgen.cli import main

    output = tmp_path / "out.urdf"
    exit_code = main(["build", str(SPECS / "bad_interface.json"), "-o", str(output)])
    assert exit_code == 1
    assert not output.exists()


# --- case 8: same port connected twice ------------------------------------


def test_case_8_double_connection_is_an_error(catalog: Catalog) -> None:
    spec = load_spec("double_connection")
    findings = validate(spec, catalog)

    assert "port_reused" in codes(errors_only(findings))


@pytest.mark.parametrize("case", INVALID_CASES, ids=lambda c: f"{c.number:02d}_{c.spec_name}")
def test_invalid_cases_are_rejected(case: Case, catalog: Catalog) -> None:
    spec: AssemblySpec = load_spec(case.spec_name)
    assert errors_only(validate(spec, catalog)) != [], f"case {case.number} was accepted"


# --- the same table, driven from text through the assembly layer ----------


def test_valid_cases_run_end_to_end_from_text(case: Case, catalog: Catalog) -> None:
    """Text in, URDF out, with the LLM scripted to answer as the table expects.

    Spec section 11 wants the acceptance cases to start from a natural-language
    request. The model is stubbed rather than mocked, so the whole path runs:
    prompt construction, schema-constrained request, response parsing,
    validation and expansion.
    """
    spec = load_spec(case.spec_name)
    with serving(replies_for(spec)) as stub:
        outcome = run_assembly(stub, catalog, case.request)

    assert outcome.succeeded, f"case {case.number} ({case.request}): {outcome.findings}"
    assert outcome.attempts == 1
    assert outcome.spec is not None
    assert [module.module for module in outcome.spec.modules] == [
        module.module for module in spec.modules
    ]

    robot = expand(outcome.spec, catalog)
    assert len(robot.movable_joints) == len(expand(spec, catalog).movable_joints)


def test_case_5_missing_module_reaches_the_user_from_text(catalog: Catalog) -> None:
    """The 'no such module' note has to survive all the way out (spec 4.4)."""
    spec = load_spec("missing_module")
    with serving(replies_for(spec)) as stub:
        outcome = run_assembly(stub, catalog, "a four-legged robot with a lidar mast")

    assert outcome.succeeded
    assert outcome.spec is not None
    assert outcome.spec.notes is not None
    assert "lidar" in outcome.spec.notes.lower()


def replies_for(spec: AssemblySpec) -> list[Reply]:
    """Script the two stages so the stub answers with ``spec``."""
    selection = {
        "name": spec.name,
        "modules": [
            {"id": module.id, "module": module.module, "mirror": module.mirror}
            for module in spec.modules
        ],
        "notes": spec.notes or "",
    }
    plan = {
        "root": spec.root,
        "connections": [
            {
                "parent": f"{connection.parent.instance}.{connection.parent.port}",
                "child": f"{connection.child.instance}.{connection.child.port}",
            }
            for connection in spec.connections
        ],
    }
    return [Reply.json_object(selection), Reply.json_object(plan)]


def run_assembly(stub: StubServer, catalog: Catalog, text: str) -> AssemblyOutcome:
    config = LLMConfig(model="stub-model", base_url=stub.base_url, timeout=10.0)
    with LocalLLMClient(config) as client:
        return assemble_with_repair(text, catalog, client)
