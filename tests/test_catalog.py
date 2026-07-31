"""Catalog schema, loader and digest tests (phase 0)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError as PydanticValidationError

from urdfgen.catalog.digest import catalog_digest, instance_port_options, module_digest
from urdfgen.catalog.loader import (
    Catalog,
    CatalogError,
    catalog_index,
    load_catalog,
    write_catalog_index,
)
from urdfgen.catalog.schema import ModuleDef

SHIPPED_MODULES = {
    "body_quad_v1",
    "leg_3dof_left_v1",
    "leg_3dof_right_v1",
    "arm_3dof_v1",
    "gripper_parallel_v1",
}

MINIMAL_MODULE: dict[str, object] = {
    "id": "probe_x_v1",
    "version": "1.0.0",
    "category": "adapter",
    "root_link": "core",
    "links": [
        {
            "name": "core",
            "visual": {"geometry": {"shape": "box", "size": [1.0, 1.0, 1.0]}},
            "inertial": {"mass": 1.0, "inertia": {"ixx": 1.0, "iyy": 1.0, "izz": 1.0}},
        }
    ],
}


def _module(**overrides: object) -> dict[str, object]:
    """A valid module definition with ``overrides`` merged in."""
    import copy

    data = copy.deepcopy(MINIMAL_MODULE)
    data.update(overrides)
    return data


# --- loading --------------------------------------------------------------


def test_shipped_catalog_loads(catalog: Catalog) -> None:
    assert set(catalog.ids()) == SHIPPED_MODULES
    assert catalog.interfaces.keys() >= {"MOUNT_A40", "TOOL_M20", "FOOT_C25"}
    assert "aluminum" in catalog.materials


def test_shipped_modules_have_declared_inertia(catalog: Catalog) -> None:
    """Spec 2.5: catalog inertia is authored, never left to be estimated."""
    for module in catalog.latest():
        for link in module.links:
            assert link.inertial is not None, f"{module.id}.{link.name} has no mass"
            assert link.inertial.inertia is not None, f"{module.id}.{link.name} has no tensor"


def test_leg_modules_are_three_dof(catalog: Catalog) -> None:
    for module_id in ("leg_3dof_left_v1", "leg_3dof_right_v1"):
        assert catalog.get(module_id).dof == 3


def test_get_resolves_latest_version(test_catalog: Catalog) -> None:
    module = test_catalog.get("probe_base_v1")
    assert module.version == "1.0.0"
    assert test_catalog.get("probe_base_v1", "1.0.0") is module


def test_get_rejects_unknown_id_and_version(catalog: Catalog) -> None:
    with pytest.raises(CatalogError, match="unknown module id"):
        catalog.get("nonexistent_v1")
    with pytest.raises(CatalogError, match="no version"):
        catalog.get("body_quad_v1", "9.9.9")


def test_has_does_not_raise(catalog: Catalog) -> None:
    """The validation layer relies on ``has`` never raising."""
    assert catalog.has("body_quad_v1")
    assert not catalog.has("nonexistent_v1")
    assert not catalog.has("body_quad_v1", "9.9.9")


def test_missing_catalog_root_is_reported(tmp_path: Path) -> None:
    with pytest.raises(CatalogError, match="does not exist"):
        load_catalog(tmp_path / "absent")


def test_missing_interfaces_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(CatalogError, match="missing"):
        load_catalog(tmp_path)


def test_module_id_must_match_directory(tmp_path: Path) -> None:
    _write_catalog(tmp_path, {"wrong_name": _module()})
    with pytest.raises(CatalogError, match="does not match its directory"):
        load_catalog(tmp_path)


def test_undeclared_interface_is_reported(tmp_path: Path) -> None:
    module = _module(ports=[{"name": "plug", "role": "male", "interface": "NOPE", "link": "core"}])
    _write_catalog(tmp_path, {"probe_x_v1": module})
    with pytest.raises(CatalogError, match=r"not declared in interfaces\.yaml"):
        load_catalog(tmp_path)


def test_undeclared_material_is_reported(tmp_path: Path) -> None:
    links = [{**MINIMAL_MODULE["links"][0], "material": "unobtainium"}]  # type: ignore[index]
    _write_catalog(tmp_path, {"probe_x_v1": _module(links=links)})
    with pytest.raises(CatalogError, match=r"not declared in materials\.yaml"):
        load_catalog(tmp_path)


def test_version_pinning_is_exact(tmp_path: Path) -> None:
    """A pinned version resolves only if that exact version is present.

    Reproducibility (spec 2.5) depends on a pin failing loudly when the
    catalog has moved on, rather than quietly resolving to something else.
    """
    _write_catalog(tmp_path, {"probe_x_v1": _module(version="1.2.0")})
    catalog = load_catalog(tmp_path)
    assert catalog.versions("probe_x_v1") == ["1.2.0"]
    assert catalog.get("probe_x_v1", "1.2.0").version == "1.2.0"
    assert not catalog.has("probe_x_v1", "1.0.0")


# --- schema ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        pytest.param({"root_link": "absent"}, "root_link", id="unknown_root_link"),
        pytest.param({"version": "1.0"}, "semver", id="bad_version"),
        pytest.param(
            {
                "links": [MINIMAL_MODULE["links"][0], MINIMAL_MODULE["links"][0]],  # type: ignore[index]
            },
            "duplicate name",
            id="duplicate_link_name",
        ),
        pytest.param(
            {"joints": [{"name": "j", "type": "fixed", "parent": "core", "child": "absent"}]},
            "unknown child link",
            id="dangling_joint",
        ),
        pytest.param(
            {"ports": [{"name": "p", "role": "male", "interface": "X", "link": "absent"}]},
            "unknown link",
            id="port_on_unknown_link",
        ),
        pytest.param(
            {
                "joints": [
                    {
                        "name": "j",
                        "type": "revolute",
                        "parent": "core",
                        "child": "core",
                    }
                ]
            },
            "to itself",
            id="self_joint",
        ),
    ],
)
def test_schema_rejects_malformed_modules(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(PydanticValidationError, match=message):
        ModuleDef.model_validate(_module(**overrides))


def test_revolute_joint_requires_limit() -> None:
    data = _module(
        links=[
            MINIMAL_MODULE["links"][0],  # type: ignore[index]
            {
                "name": "second",
                "visual": {"geometry": {"shape": "box", "size": [1.0, 1.0, 1.0]}},
                "inertial": {"mass": 1.0, "inertia": {"ixx": 1.0, "iyy": 1.0, "izz": 1.0}},
            },
        ],
        joints=[{"name": "j", "type": "revolute", "parent": "core", "child": "second"}],
    )
    with pytest.raises(PydanticValidationError, match="requires a 'limit'"):
        ModuleDef.model_validate(data)


def test_unreachable_link_is_rejected() -> None:
    data = _module(
        links=[
            MINIMAL_MODULE["links"][0],  # type: ignore[index]
            {
                "name": "floating",
                "visual": {"geometry": {"shape": "box", "size": [1.0, 1.0, 1.0]}},
                "inertial": {"mass": 1.0, "inertia": {"ixx": 1.0, "iyy": 1.0, "izz": 1.0}},
            },
        ]
    )
    with pytest.raises(PydanticValidationError, match="not reachable"):
        ModuleDef.model_validate(data)


def test_unknown_field_is_rejected() -> None:
    """Typos in hand-written YAML must fail loudly rather than be ignored."""
    with pytest.raises(PydanticValidationError, match="Extra inputs"):
        ModuleDef.model_validate(_module(rootlink="core"))


def test_mesh_link_must_declare_inertia() -> None:
    data = _module(
        links=[
            {
                "name": "core",
                "visual": {"geometry": {"shape": "mesh", "mesh": "meshes/core.dae"}},
                "inertial": {"mass": 1.0},
            }
        ]
    )
    with pytest.raises(PydanticValidationError, match="never estimated from meshes"):
        ModuleDef.model_validate(data)


def test_primitive_link_may_omit_the_tensor(test_catalog: Catalog) -> None:
    tip = test_catalog.get("probe_arm_v1").link("tip")
    assert tip is not None
    assert tip.inertial is not None
    assert tip.inertial.inertia is None


# --- index and digest -----------------------------------------------------


def test_catalog_index_lists_every_module(catalog: Catalog) -> None:
    index = catalog_index(catalog)
    assert {entry["id"] for entry in index["modules"]} == SHIPPED_MODULES


def test_write_catalog_index(catalog: Catalog, tmp_path: Path) -> None:
    path = write_catalog_index(catalog, tmp_path / "catalog.yaml")
    written = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert {entry["id"] for entry in written["modules"]} == SHIPPED_MODULES


def test_digest_lists_modules_and_ports(catalog: Catalog) -> None:
    digest = catalog_digest(catalog)
    assert "### body_quad_v1 (category: body)" in digest
    assert "leg_mount_fl [female/MOUNT_A40]" in digest
    for module_id in SHIPPED_MODULES:
        assert module_id in digest


def test_digest_omits_geometry_and_inertia(catalog: Catalog) -> None:
    """Spec 4.2: the LLM must not see anything it could copy as geometry."""
    digest = catalog_digest(catalog)
    for banned in ("ixx", "inertia", "meshes/", "package://", "rpy", "root_link"):
        assert banned not in digest


def test_digest_can_be_narrowed(catalog: Catalog) -> None:
    digest = catalog_digest(catalog, module_ids=["body_quad_v1"])
    assert "body_quad_v1" in digest
    assert "leg_3dof_left_v1" not in digest


def test_module_digest_reports_dof_and_mass(catalog: Catalog) -> None:
    lines = module_digest(catalog.get("leg_3dof_left_v1"))
    assert any("3 DOF" in line and "1.27 kg" in line for line in lines)


def test_instance_port_options(catalog: Catalog) -> None:
    """These strings become the enum the LLM's port references are limited to."""
    options = instance_port_options(catalog, {"body": "body_quad_v1", "leg_fl": "leg_3dof_left_v1"})
    assert "body.leg_mount_fl" in options
    assert "leg_fl.base" in options
    assert "leg_fl.nonexistent" not in options


# --- helpers --------------------------------------------------------------


def _write_catalog(root: Path, modules: dict[str, dict[str, object]]) -> None:
    """Lay out a throwaway catalog: interfaces, materials and module files."""
    (root / "interfaces.yaml").write_text(
        yaml.safe_dump({"interfaces": [{"id": "TEST_A"}]}), encoding="utf-8"
    )
    (root / "materials.yaml").write_text(
        yaml.safe_dump({"materials": [{"id": "plain", "rgba": [0.5, 0.5, 0.5, 1.0]}]}),
        encoding="utf-8",
    )
    for directory, data in modules.items():
        (root / directory).mkdir(parents=True, exist_ok=True)
        (root / directory / "module.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
