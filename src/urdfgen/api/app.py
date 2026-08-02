"""The HTTP API, and the process that serves the web app.

The routes are a transport over the same pure functions the CLI calls. No check
is re-implemented here and no XML is assembled here: ``/api/validate`` runs
:func:`~urdfgen.validation.validate` and ``/api/build`` runs
:func:`~urdfgen.builder.expand.expand` plus
:func:`~urdfgen.builder.urdf.build_urdf_string`, so a spec that builds over HTTP
builds identically from the command line.

That is what keeps the browser honest. The web app can draw an assembly and
decide, live, whether a part fits a hardpoint -- interface, role, occupancy --
because that has to be interactive. Everything else it asks this server for. It
never writes URDF, for the same reason the LLM never does.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from urdfgen.api.models import (
    AssembleRequest,
    AssembleResponse,
    BuildRequest,
    BuildResponse,
    DefaultAssembly,
    Health,
    LLMStatus,
    Report,
    SpecRequest,
    Summary,
)
from urdfgen.builder.expand import ExpansionError, expand
from urdfgen.builder.urdf import build_urdf_string
from urdfgen.catalog.loader import Catalog, load_catalog
from urdfgen.catalog.payload import PAYLOAD_VERSION, catalog_payload
from urdfgen.llm.client import LLMError, LocalLLMClient
from urdfgen.llm.config import LLMConfig, LLMConfigError, load_config
from urdfgen.llm.repair import assemble_with_repair
from urdfgen.ui.page import starting_assembly
from urdfgen.validation import has_errors, validate

TITLE = "urdfgen"
API_PREFIX = "/api"

_NO_WEB_APP = """<!doctype html>
<title>urdfgen</title>
<style>body{font:14px/1.6 ui-monospace,monospace;margin:3rem auto;max-width:34rem}</style>
<h1>urdfgen</h1>
<p>The API is running. The web app has not been built into this checkout yet:</p>
<pre>cd web &amp;&amp; npm install &amp;&amp; npm run build</pre>
<p>Then restart the server, or point it at the build with
<code>urdfgen serve --web path/to/dist</code>.</p>
<p>The API itself is at <a href="/api/health">/api/health</a>, and
<code>urdfgen ui -o assembly.html</code> still writes the standalone page.</p>
"""


def default_web_root() -> Path:
    """Where ``npm run build`` puts the web app in this repository."""
    return Path(__file__).resolve().parents[3] / "web" / "dist"


def create_app(
    catalog: Catalog | None = None,
    *,
    web_root: Path | None = None,
) -> FastAPI:
    """Build the application.

    ``catalog`` is read once at startup and held for the life of the process:
    it is immutable, and re-reading it per request would let two requests in one
    session disagree about what the catalog contains.
    """
    loaded = catalog if catalog is not None else load_catalog()
    site = web_root if web_root is not None else default_web_root()
    has_site = (site / "index.html").is_file()

    app = FastAPI(
        title=TITLE,
        summary="Assemble URDF robot descriptions from a catalog of modules.",
        version=str(PAYLOAD_VERSION),
    )
    app.state.catalog = loaded

    @app.get(f"{API_PREFIX}/health", response_model=Health)
    def health() -> Health:
        return Health(
            payload_version=PAYLOAD_VERSION,
            modules=len(loaded.modules),
            catalog_root=str(loaded.root) if loaded.root else None,
            web_app=has_site,
            llm=_llm_status(),
        )

    @app.get(f"{API_PREFIX}/catalog")
    def catalog_route() -> dict[str, Any]:
        """Every module, with the geometry and port frames needed to draw it."""
        return catalog_payload(loaded)

    @app.get(f"{API_PREFIX}/assembly/default", response_model=DefaultAssembly)
    def default_assembly() -> DefaultAssembly:
        """The unit the web app opens on: a body with every limb mount filled."""
        return DefaultAssembly(spec=starting_assembly(loaded))

    @app.post(f"{API_PREFIX}/validate", response_model=Report)
    def validate_route(request: SpecRequest) -> Report:
        """Run the whole validation layer. Findings are data, never a failure."""
        return Report.of(validate(request.spec, loaded))

    @app.post(f"{API_PREFIX}/build", response_model=BuildResponse)
    def build_route(request: BuildRequest) -> BuildResponse:
        """Turn a spec into URDF, refusing one that failed validation."""
        findings = validate(request.spec, loaded)
        if has_errors(findings) and not request.skip_validation:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "validation_failed",
                    "message": "the assembly has errors, so no URDF was built",
                    "report": Report.of(findings).model_dump(),
                },
            )

        try:
            robot = expand(request.spec, loaded)
        except ExpansionError as exc:
            raise HTTPException(
                status_code=422,
                detail={"error": "expansion_failed", "message": str(exc)},
            ) from exc

        return BuildResponse(
            urdf=build_urdf_string(
                robot,
                loaded,
                package_name=request.package_name,
                source_text=request.spec.name,
            ),
            summary=Summary.of(robot),
            report=Report.of(findings),
        )

    @app.post(f"{API_PREFIX}/assemble", response_model=AssembleResponse)
    def assemble_route(request: AssembleRequest) -> AssembleResponse:
        """Natural language to ``AssemblySpec``, through the local model.

        The one route that needs something outside this process. It returns the
        spec even when validation failed -- an editable starting point beats an
        error message -- so callers must read ``succeeded``, not just the status.
        """
        try:
            config = load_config()
        except LLMConfigError as exc:
            raise HTTPException(
                status_code=503,
                detail={"error": "llm_unavailable", "message": str(exc)},
            ) from exc

        try:
            with LocalLLMClient(config) as client:
                outcome = assemble_with_repair(
                    request.text, loaded, client, max_attempts=request.max_attempts
                )
        except LLMError as exc:
            raise HTTPException(
                status_code=502,
                detail={"error": "llm_failed", "message": str(exc)},
            ) from exc

        return AssembleResponse.of(outcome)

    if has_site:
        # Mounted last: a path that matches a route above is never shadowed by a
        # file that happens to share its name.
        app.mount("/", StaticFiles(directory=site, html=True), name="web")
    else:

        @app.get("/", response_class=HTMLResponse, include_in_schema=False)
        def placeholder() -> str:
            return _NO_WEB_APP

    return app


def _llm_status() -> LLMStatus:
    """Whether ``/api/assemble`` can work, without calling the model to find out."""
    try:
        config: LLMConfig = load_config()
    except LLMConfigError as exc:
        return LLMStatus(configured=False, detail=str(exc))
    return LLMStatus(configured=True, model=config.model, base_url=config.base_url)
