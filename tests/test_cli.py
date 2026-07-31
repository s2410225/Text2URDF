"""Command line behaviour, including exit codes."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
import yourdfpy

from conftest import SPECS
from urdfgen.cli import EXIT_FAILED, EXIT_OK, EXIT_USAGE, main


def test_list_prints_every_category(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["list"]) == EXIT_OK
    output = capsys.readouterr().out
    for category in ("body:", "leg:", "arm:", "gripper:"):
        assert category in output


def test_list_can_filter_by_category(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["list", "--category", "gripper"]) == EXIT_OK
    output = capsys.readouterr().out
    assert "gripper_parallel_v1" in output
    assert "body_quad_v1" not in output


def test_show_reports_ports_and_dof(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["show", "leg_3dof_left_v1"]) == EXIT_OK
    output = capsys.readouterr().out
    assert "base [male/MOUNT_A40] on hip_bracket" in output
    assert "dof: 3" in output


def test_show_unknown_module_fails(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["show", "not_a_module_v1"]) == EXIT_FAILED
    assert "unknown module id" in capsys.readouterr().err


def test_build_module_writes_parsable_urdf(tmp_path: Path) -> None:
    """The phase 0 completion criterion, exercised end to end."""
    output = tmp_path / "leg.urdf"
    assert main(["build-module", "leg_3dof_left_v1", "-o", str(output)]) == EXIT_OK

    parsed = yourdfpy.URDF.load(
        io.BytesIO(output.read_bytes()),
        build_scene_graph=False,
        build_collision_scene_graph=False,
        load_meshes=False,
        load_collision_meshes=False,
    )
    assert len(parsed.link_map) == 4
    assert len(parsed.joint_map) == 3


def test_build_module_writes_to_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["build-module", "body_quad_v1"]) == EXIT_OK
    assert '<robot name="body_quad_v1">' in capsys.readouterr().out


def test_build_writes_the_quadruped(tmp_path: Path) -> None:
    """The phase 1 completion criterion."""
    output = tmp_path / "quad.urdf"
    assert main(["build", str(SPECS / "quadruped_a.json"), "-o", str(output)]) == EXIT_OK

    parsed = yourdfpy.URDF.load(
        io.BytesIO(output.read_bytes()),
        build_scene_graph=False,
        build_collision_scene_graph=False,
        load_meshes=False,
        load_collision_meshes=False,
    )
    assert len(parsed.link_map) == 17
    assert sum(1 for joint in parsed.joint_map.values() if joint.type != "fixed") == 12


def test_build_uses_the_requested_package_name(tmp_path: Path) -> None:
    output = tmp_path / "quad.urdf"
    assert (
        main(
            [
                "build",
                str(SPECS / "quadruped_a.json"),
                "-o",
                str(output),
                "--package",
                "my_robot_description",
            ]
        )
        == EXIT_OK
    )
    assert output.exists()


def test_validate_succeeds_on_a_good_spec(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["validate", str(SPECS / "quadruped_a.json")]) == EXIT_OK
    assert "0 error(s)" in capsys.readouterr().out


def test_validate_fails_on_a_bad_spec(capsys: pytest.CaptureFixture[str]) -> None:
    """The phase 2 completion criterion: errors mean a non-zero exit."""
    assert main(["validate", str(SPECS / "bad_interface.json")]) == EXIT_FAILED
    assert "interface_mismatch" in capsys.readouterr().out


def test_build_refuses_an_invalid_spec(tmp_path: Path) -> None:
    output = tmp_path / "out.urdf"
    assert main(["build", str(SPECS / "bad_interface.json"), "-o", str(output)]) == EXIT_FAILED
    assert not output.exists()


def test_build_can_be_forced_past_validation(tmp_path: Path) -> None:
    """An escape hatch for inspecting what a rejected spec would produce."""
    output = tmp_path / "out.urdf"
    exit_code = main(
        ["build", str(SPECS / "bad_interface.json"), "-o", str(output), "--skip-validation"]
    )
    assert exit_code == EXIT_OK
    assert output.exists()


def test_malformed_spec_file_is_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    assert main(["validate", str(path)]) == EXIT_USAGE
    assert "not valid JSON" in capsys.readouterr().err


def test_spec_failing_the_ir_schema_is_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "wrong.json"
    path.write_text('{"name": "x", "root": "a", "modules": []}', encoding="utf-8")
    assert main(["validate", str(path)]) == EXIT_USAGE
    assert "not a valid AssemblySpec" in capsys.readouterr().err


def test_missing_catalog_is_reported(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--catalog", str(tmp_path / "absent"), "list"]) == EXIT_FAILED
    assert "does not exist" in capsys.readouterr().err


def test_index_writes_the_generated_catalog(tmp_path: Path) -> None:
    import shutil

    from conftest import REPO_ROOT

    catalog_root = tmp_path / "modules"
    shutil.copytree(REPO_ROOT / "modules", catalog_root)
    assert main(["--catalog", str(catalog_root), "index"]) == EXIT_OK
    assert (catalog_root / "catalog.yaml").is_file()


def test_digest_prints_the_llm_summary(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["digest"]) == EXIT_OK
    output = capsys.readouterr().out
    assert "## Available modules" in output
    assert "ixx" not in output
