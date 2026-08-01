"""Build the standalone assembly page from a catalog.

Everything the page needs is injected as JSON, so the template carries no data
of its own and cannot go stale when the catalog changes.
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any

from urdfgen.catalog.loader import Catalog
from urdfgen.catalog.payload import catalog_payload
from urdfgen.ir.assembly import AssemblySpec, Connection, ModuleInstance, PortRef

TEMPLATE = "assembly.html"
DEFAULT_TITLE = "urdfgen — assembly"
DEFAULT_UNIT_NAME = "unit_01"

_PLACEHOLDERS = ("__TITLE__", "__PAYLOAD__", "__ASSEMBLY__", "__MODULE_COUNT__")


class UIError(Exception):
    """Raised when a page cannot be generated from the given catalog."""


def render_page(
    catalog: Catalog,
    *,
    title: str = DEFAULT_TITLE,
    assembly: AssemblySpec | None = None,
) -> str:
    """Return the complete, self-contained HTML page."""
    payload = catalog_payload(catalog)
    if not payload["modules"]:
        msg = "the catalog has no modules, so there is nothing to assemble"
        raise UIError(msg)

    start = assembly or starting_assembly(catalog)
    template = _template()

    page = template
    for placeholder, value in (
        ("__TITLE__", title),
        ("__MODULE_COUNT__", str(len(payload["modules"]))),
        ("__PAYLOAD__", _json(payload)),
        ("__ASSEMBLY__", _json(start.model_dump(exclude_none=True))),
    ):
        page = page.replace(placeholder, value)

    left = [name for name in _PLACEHOLDERS if name in page]
    if left:
        msg = f"template placeholders were not filled: {', '.join(left)}"
        raise UIError(msg)
    return page


def write_page(
    catalog: Catalog,
    output: str | Path,
    *,
    title: str = DEFAULT_TITLE,
    assembly: AssemblySpec | None = None,
) -> Path:
    path = Path(output)
    path.write_text(render_page(catalog, title=title, assembly=assembly), encoding="utf-8")
    return path


def starting_assembly(catalog: Catalog) -> AssemblySpec:
    """The unit the page opens with.

    A body with every limb mount populated, so the first thing on screen is a
    whole machine rather than a lone chassis. Built by the same rules the page
    itself uses -- interface match and opposite roles -- so it stays valid if
    the catalog changes.
    """
    frame = _first(catalog, "body") or _first(catalog, "adapter")
    if frame is None:
        msg = "the catalog has no body module to build an assembly around"
        raise UIError(msg)

    modules = [ModuleInstance(id="body", module=frame.id, version=frame.version)]
    connections: list[Connection] = []

    for port in frame.ports:
        if port.interface != "MOUNT_A40":
            continue
        side = "right" if port.name.endswith(("_fr", "_rr")) else "left"
        limb = _first(catalog, "leg", prefer=side)
        if limb is None:
            continue
        mount = next(
            (
                candidate
                for candidate in limb.ports
                if candidate.link == limb.root_link
                and candidate.interface == port.interface
                and candidate.role != port.role
            ),
            None,
        )
        if mount is None:
            continue

        instance = f"leg_{port.name.rsplit('_', 1)[-1]}"
        modules.append(ModuleInstance(id=instance, module=limb.id, version=limb.version))
        connections.append(
            Connection(
                parent=PortRef(instance="body", port=port.name),
                child=PortRef(instance=instance, port=mount.name),
            )
        )

    return AssemblySpec(
        name=DEFAULT_UNIT_NAME,
        root="body",
        modules=modules,
        connections=connections,
        notes=None,
    )


def _first(catalog: Catalog, category: str, prefer: str | None = None) -> Any:
    """The first module of a category, favouring one whose id contains ``prefer``."""
    candidates = [module for module in catalog.latest() if module.category == category]
    if prefer:
        for module in candidates:
            if prefer in module.id:
                return module
    return candidates[0] if candidates else None


def _template() -> str:
    return (resources.files("urdfgen.ui.assets") / TEMPLATE).read_text(encoding="utf-8")


def _json(value: object) -> str:
    """Compact JSON that is safe to paste inside a <script> block."""
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=False)
    return text.replace("</", "<\\/")
