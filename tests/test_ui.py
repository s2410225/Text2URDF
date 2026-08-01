"""Catalog payload and assembly page tests.

How the page looks is not something a test can judge. What it can check is that
the data reaching the browser is the catalog -- all of it, unaltered -- that the
page carries that data rather than a stale copy, and that it depends on nothing
it cannot reach.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from urdfgen.catalog.loader import Catalog
from urdfgen.catalog.payload import PAYLOAD_VERSION, catalog_payload, module_payload
from urdfgen.catalog.schema import MeshGeometry
from urdfgen.ui import render_page, write_page
from urdfgen.ui.page import UIError, starting_assembly
from urdfgen.validation import errors_only, validate

# --- payload --------------------------------------------------------------


def test_payload_carries_every_module(catalog: Catalog) -> None:
    payload = catalog_payload(catalog)
    assert [module["id"] for module in payload["modules"]] == catalog.ids()
    assert payload["version"] == PAYLOAD_VERSION


def test_payload_figures_match_the_catalog(catalog: Catalog) -> None:
    for entry in catalog_payload(catalog)["modules"]:
        module = catalog.get(entry["id"])
        assert entry["mass"] == pytest.approx(module.total_mass)
        assert entry["dof"] == module.dof
        assert entry["version"] == module.version
        assert entry["root_link"] == module.root_link


def test_payload_carries_every_port_with_its_pairing_rule(catalog: Catalog) -> None:
    """Interface and role are what decide whether a part can be fitted."""
    for entry in catalog_payload(catalog)["modules"]:
        module = catalog.get(entry["id"])
        assert [(p["name"], p["role"], p["interface"]) for p in entry["ports"]] == [
            (p.name, p.role, p.interface) for p in module.ports
        ]


def test_payload_carries_geometry_for_every_visual(catalog: Catalog) -> None:
    """A client has to place parts, not just list them."""
    for entry in catalog_payload(catalog)["modules"]:
        module = catalog.get(entry["id"])
        for link, source in zip(entry["links"], module.links, strict=True):
            assert link["name"] == source.name
            if source.visual is None:
                assert link["visual"] is None
            else:
                assert link["visual"]["geometry"]["shape"] == source.visual.geometry.shape
                assert link["visual"]["origin"]["xyz"] == list(source.visual.origin.xyz)


def test_payload_carries_joint_limits(catalog: Catalog) -> None:
    """Without limits the page cannot bound its sliders."""
    leg = next(m for m in catalog_payload(catalog)["modules"] if m["id"] == "leg_3dof_left_v1")
    knee = next(joint for joint in leg["joints"] if joint["name"] == "knee_pitch")
    assert knee["movable"] is True
    assert knee["limit"]["lower"] == pytest.approx(-2.36)
    assert knee["limit"]["upper"] == pytest.approx(0.0)


def test_payload_carries_material_colours(catalog: Catalog) -> None:
    payload = catalog_payload(catalog)
    assert set(payload["materials"]) == set(catalog.materials)
    assert payload["materials"]["accent_orange"][:3] == pytest.approx([0.88, 0.42, 0.12])


def test_payload_is_json_serialisable(catalog: Catalog) -> None:
    """It travels as JSON, so nothing exotic may sneak into it."""
    assert json.loads(json.dumps(catalog_payload(catalog)))


def test_module_payload_reports_a_mesh_rather_than_dropping_it(catalog: Catalog) -> None:
    """A client that cannot draw a mesh still needs to know one is there."""
    module = catalog.get("body_quad_v1")
    visual = module.links[0].visual
    assert visual is not None
    swapped = module.model_copy(
        update={
            "links": (
                module.links[0].model_copy(
                    update={
                        "visual": visual.model_copy(
                            update={"geometry": MeshGeometry(mesh="meshes/x.stl")}
                        )
                    }
                ),
            )
        }
    )
    geometry = module_payload(swapped)["links"][0]["visual"]["geometry"]
    assert geometry["shape"] == "mesh"
    assert geometry["mesh"] == "meshes/x.stl"


# --- starting assembly ----------------------------------------------------


def test_the_page_opens_on_a_whole_machine(catalog: Catalog) -> None:
    spec = starting_assembly(catalog)
    assert spec.root == "body"
    assert len(spec.modules) == 5
    assert len(spec.connections) == 4


def test_the_starting_assembly_is_valid(catalog: Catalog) -> None:
    """The first thing on screen must not already be broken."""
    assert errors_only(validate(starting_assembly(catalog), catalog)) == []


def test_the_starting_assembly_gets_the_handedness_right(catalog: Catalog) -> None:
    spec = starting_assembly(catalog)
    by_id = {instance.id: instance.module for instance in spec.modules}
    assert by_id["leg_fl"] == "leg_3dof_left_v1"
    assert by_id["leg_fr"] == "leg_3dof_right_v1"
    assert by_id["leg_rl"] == "leg_3dof_left_v1"
    assert by_id["leg_rr"] == "leg_3dof_right_v1"


def test_versions_are_pinned(catalog: Catalog) -> None:
    for instance in starting_assembly(catalog).modules:
        assert instance.version == catalog.get(instance.module).version


# --- the page -------------------------------------------------------------


def test_page_is_self_contained(catalog: Catalog) -> None:
    """No CDN, no build step: a published artifact can fetch nothing."""
    page = render_page(catalog)
    assert "http://" not in page
    assert "https://" not in page.replace('xmlns="http://www.w3.org/2000/svg"', "")
    assert not re.search(r"<script[^>]+src=", page)
    assert not re.search(r"<link[^>]+href=", page)
    assert "@import" not in page


def test_page_leaves_no_placeholders(catalog: Catalog) -> None:
    page = render_page(catalog)
    assert "__PAYLOAD__" not in page
    assert "__ASSEMBLY__" not in page
    assert "__TITLE__" not in page


def test_page_carries_the_whole_catalog(catalog: Catalog) -> None:
    """The data is injected, never hand-copied, so it cannot go stale."""
    page = render_page(catalog)
    for module_id in catalog.ids():
        assert module_id in page
    for interface in catalog.interfaces:
        assert interface in page


def test_page_embeds_a_parsable_payload(catalog: Catalog) -> None:
    page = render_page(catalog)
    match = re.search(r"const CATALOG = (\{.*?\});\n", page, re.S)
    assert match is not None
    payload = json.loads(match.group(1).replace("<\\/", "</"))
    assert [module["id"] for module in payload["modules"]] == catalog.ids()


def test_page_embeds_the_starting_assembly(catalog: Catalog) -> None:
    page = render_page(catalog)
    match = re.search(r"const DEFAULT_ASSEMBLY = (\{.*?\});\n", page, re.S)
    assert match is not None
    assert json.loads(match.group(1).replace("<\\/", "</"))["root"] == "body"


def test_closing_tags_in_data_cannot_break_out_of_the_script(catalog: Catalog) -> None:
    """Catalog text is authored, but it still must not end the script block."""
    module = catalog.get("body_quad_v1")
    hostile = module.model_copy(update={"description": "</script><b>escaped</b>"})
    page = render_page(
        Catalog(
            modules={**catalog.modules, "body_quad_v1": {module.version: hostile}},
            interfaces=catalog.interfaces,
            materials=catalog.materials,
            root=catalog.root,
        )
    )
    assert "</script><b>" not in page
    assert "<\\/script>" in page


def test_every_element_the_script_reaches_for_exists(catalog: Catalog) -> None:
    """A renamed id would break a panel silently, in the browser, at runtime.

    The script and the markup live in one file but nothing checks that they
    agree, so check it here.
    """
    page = render_page(catalog)
    wanted = set(re.findall(r'getElementById\("([^"]+)"\)', page))
    assert wanted, "expected the page to look elements up by id"

    declared = set(re.findall(r'\bid="([^"]+)"', page))
    assert wanted <= declared, f"script reaches for missing ids: {sorted(wanted - declared)}"


def test_page_uses_the_requested_title(catalog: Catalog) -> None:
    assert "<title>GARAGE</title>" in render_page(catalog, title="GARAGE")


def test_page_is_deterministic(catalog: Catalog) -> None:
    assert render_page(catalog) == render_page(catalog)


def test_write_page(catalog: Catalog, tmp_path: Path) -> None:
    path = write_page(catalog, tmp_path / "assembly.html")
    assert path.read_text(encoding="utf-8").startswith("<title>")


def test_an_empty_catalog_is_refused(tmp_path: Path) -> None:
    with pytest.raises(UIError, match="no modules"):
        render_page(Catalog(modules={}, interfaces={}, materials={}, root=tmp_path))


def test_a_catalog_without_a_body_is_refused(catalog: Catalog) -> None:
    legs = {name: versions for name, versions in catalog.modules.items() if "leg" in name}
    with pytest.raises(UIError, match="no body module"):
        render_page(
            Catalog(
                modules=legs,
                interfaces=catalog.interfaces,
                materials=catalog.materials,
                root=catalog.root,
            )
        )
