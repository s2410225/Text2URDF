"""HTTP API tests (phase 4).

The API is a transport, so what these check is that it transports faithfully:
that the catalog on the wire is the catalog, that findings are the validation
layer's own findings, that a spec with errors gets no URDF, and that the XML it
returns is byte for byte what ``urdfgen build`` writes. A route that quietly
re-decided any of those would be the failure worth catching.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from conftest import load_spec
from stub_llm import StubServer, serving
from test_llm import quad_replies
from urdfgen.api import API_PREFIX, create_app
from urdfgen.builder.expand import expand
from urdfgen.builder.urdf import build_urdf_string
from urdfgen.catalog.loader import Catalog
from urdfgen.catalog.payload import PAYLOAD_VERSION, catalog_payload
from urdfgen.catalog.schema import MeshGeometry
from urdfgen.ir.assembly import AssemblySpec
from urdfgen.ui.page import starting_assembly
from urdfgen.validation import validate


@pytest.fixture
def client(catalog: Catalog, tmp_path: Path) -> Iterator[TestClient]:
    """A client for an app with no web app built, so ``/`` is the placeholder."""
    with TestClient(create_app(catalog, web_root=tmp_path / "absent")) as running:
        yield running


def post(client: TestClient, route: str, **body: Any) -> Any:
    return client.post(f"{API_PREFIX}{route}", json=body)


def spec_body(spec: AssemblySpec) -> Any:
    return json.loads(spec.model_dump_json(exclude_none=True))


# --- catalog --------------------------------------------------------------


def test_catalog_route_returns_the_catalog_payload(client: TestClient, catalog: Catalog) -> None:
    """The route adds nothing of its own; the payload function is the contract."""
    response = client.get(f"{API_PREFIX}/catalog")
    assert response.status_code == 200
    assert response.json() == json.loads(json.dumps(catalog_payload(catalog)))


def test_health_reports_what_the_process_can_do(
    client: TestClient, catalog: Catalog, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("LLM_MODEL", raising=False)
    body = client.get(f"{API_PREFIX}/health").json()
    assert body["status"] == "ok"
    assert body["payload_version"] == PAYLOAD_VERSION
    assert body["modules"] == len(catalog.modules)
    assert body["web_app"] is False
    assert body["llm"]["configured"] is False
    assert "LLM_MODEL" in body["llm"]["detail"]


def test_health_reports_a_configured_model_without_calling_it(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LLM_MODEL", "qwen2.5:7b")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:9/v1")
    llm = client.get(f"{API_PREFIX}/health").json()["llm"]
    assert llm == {
        "configured": True,
        "model": "qwen2.5:7b",
        "base_url": "http://localhost:9/v1",
        "detail": None,
    }


def test_the_default_assembly_is_the_one_the_page_opens_on(
    client: TestClient, catalog: Catalog
) -> None:
    body = client.get(f"{API_PREFIX}/assembly/default").json()
    assert AssemblySpec.model_validate(body["spec"]) == starting_assembly(catalog)


# --- validate -------------------------------------------------------------


def test_validate_returns_the_validation_layer_findings(
    client: TestClient, catalog: Catalog
) -> None:
    spec = load_spec("swapped_handedness")
    body = post(client, "/validate", spec=spec_body(spec)).json()
    expected = validate(spec, catalog)
    assert [finding["code"] for finding in body["findings"]] == [f.code for f in expected]
    assert body["errors"] == sum(1 for f in expected if f.is_error)
    assert body["warnings"] == len(expected) - body["errors"]


def test_a_valid_assembly_validates_clean(client: TestClient) -> None:
    body = post(client, "/validate", spec=spec_body(load_spec("quadruped_a"))).json()
    assert body["ok"] is True
    assert body["errors"] == 0


def test_an_invalid_assembly_is_reported_not_raised(client: TestClient) -> None:
    """Findings are data. A broken assembly is a 200 with errors in it."""
    response = post(client, "/validate", spec=spec_body(load_spec("bad_interface")))
    assert response.status_code == 200
    assert response.json()["ok"] is False


def test_a_spec_that_is_not_an_assembly_is_refused(client: TestClient) -> None:
    """``extra='forbid'`` reaches the HTTP boundary, so nothing is dropped."""
    body = spec_body(load_spec("quadruped_a")) | {"colour": "red"}
    assert post(client, "/validate", spec=body).status_code == 422


# --- build ----------------------------------------------------------------


def test_build_returns_exactly_what_the_cli_writes(client: TestClient, catalog: Catalog) -> None:
    """One builder, two front ends. A divergence here would be a second builder."""
    spec = load_spec("quadruped_arm_gripper")
    body = post(client, "/build", spec=spec_body(spec)).json()
    robot = expand(spec, catalog)
    assert body["urdf"] == build_urdf_string(robot, catalog, source_text=spec.name)


def test_build_summarises_the_robot_it_built(client: TestClient, catalog: Catalog) -> None:
    spec = load_spec("quadruped_arm_gripper")
    summary = post(client, "/build", spec=spec_body(spec)).json()["summary"]
    robot = expand(spec, catalog)
    assert summary["links"] == len(robot.links)
    assert summary["joints"] == len(robot.joints)
    assert summary["movable_joints"] == len(robot.movable_joints)
    assert summary["mass"] == pytest.approx(robot.total_mass)
    assert summary["modules_used"][0]["version"]


def test_build_refuses_an_assembly_with_errors(client: TestClient) -> None:
    response = post(client, "/build", spec=spec_body(load_spec("bad_interface")))
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["error"] == "validation_failed"
    assert detail["report"]["errors"] > 0
    assert "urdf" not in response.text


def test_build_can_be_forced_past_validation(client: TestClient) -> None:
    """The same escape hatch the CLI has, and the findings still come back."""
    response = post(
        client, "/build", spec=spec_body(load_spec("bad_interface")), skip_validation=True
    )
    assert response.status_code == 200
    assert response.json()["report"]["ok"] is False
    assert response.json()["urdf"].startswith("<?xml")


def test_build_honours_the_package_name(catalog: Catalog, tmp_path: Path) -> None:
    """The package name only shows up in mesh URIs, so give the catalog a mesh.

    The shipped modules are primitives, which would make this pass whatever the
    route did with the parameter.
    """
    body = catalog.get("body_quad_v1")
    visual = body.links[0].visual
    assert visual is not None
    meshed = body.model_copy(
        update={
            "links": (
                body.links[0].model_copy(
                    update={
                        "visual": visual.model_copy(
                            update={"geometry": MeshGeometry(mesh="meshes/chassis.stl")}
                        )
                    }
                ),
                *body.links[1:],
            )
        }
    )
    with_mesh = Catalog(
        modules={**catalog.modules, "body_quad_v1": {body.version: meshed}},
        interfaces=catalog.interfaces,
        materials=catalog.materials,
        root=catalog.root,
    )

    spec = spec_body(load_spec("quadruped_a"))
    with TestClient(create_app(with_mesh, web_root=tmp_path / "absent")) as client:
        urdf = post(client, "/build", spec=spec, package_name="my_robot").json()["urdf"]
    assert "package://my_robot/" in urdf


# --- assemble -------------------------------------------------------------


@pytest.fixture
def stub() -> Iterator[StubServer]:
    with serving(quad_replies()) as state:
        yield state


def test_assemble_needs_a_configured_model(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("LLM_MODEL", raising=False)
    response = post(client, "/assemble", text="4脚ロボットを作って")
    assert response.status_code == 503
    assert response.json()["detail"]["error"] == "llm_unavailable"


def test_assemble_runs_the_repair_loop(
    client: TestClient, stub: StubServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Text in, ``AssemblySpec`` out -- and no URDF, which is the whole point."""
    monkeypatch.setenv("LLM_MODEL", "stub-model")
    monkeypatch.setenv("LLM_BASE_URL", stub.base_url)

    body = post(client, "/assemble", text="4脚ロボットを作って").json()
    assert body["succeeded"] is True
    assert body["report"]["errors"] == 0
    assert AssemblySpec.model_validate(body["spec"]).root == "body"
    assert "urdf" not in body


def test_assemble_reports_an_unreachable_model(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LLM_MODEL", "stub-model")
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:1/v1")
    response = post(client, "/assemble", text="4脚ロボット")
    assert response.status_code == 502
    assert response.json()["detail"]["error"] == "llm_failed"


def test_assemble_needs_something_to_assemble(client: TestClient) -> None:
    assert post(client, "/assemble", text="").status_code == 422


# --- the web app ----------------------------------------------------------


def test_without_a_built_web_app_the_root_says_how_to_build_it(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "npm run build" in response.text


def test_a_built_web_app_is_served_at_the_root(catalog: Catalog, tmp_path: Path) -> None:
    site = tmp_path / "dist"
    site.mkdir()
    (site / "index.html").write_text("<title>web</title>", encoding="utf-8")

    with TestClient(create_app(catalog, web_root=site)) as client:
        assert client.get("/").text == "<title>web</title>"
        assert client.get(f"{API_PREFIX}/health").json()["web_app"] is True


def test_the_web_app_never_shadows_the_api(catalog: Catalog, tmp_path: Path) -> None:
    """Static files are mounted last, so a stray ``api/`` in the build is inert."""
    site = tmp_path / "dist"
    (site / "api").mkdir(parents=True)
    (site / "index.html").write_text("<title>web</title>", encoding="utf-8")
    (site / "api" / "health").write_text("not the api", encoding="utf-8")

    with TestClient(create_app(catalog, web_root=site)) as client:
        assert client.get(f"{API_PREFIX}/health").json()["status"] == "ok"
