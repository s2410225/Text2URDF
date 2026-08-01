"""The catalog as plain JSON, for a UI to consume.

:func:`catalog_index` in :mod:`urdfgen.catalog.loader` summarises the catalog
for humans; this carries enough to *draw* it -- link geometry, joint frames and
port frames -- because a viewer has to place parts, not just list them.

This is the payload ``GET /api/catalog`` will return in phase 4, so the shape is
chosen for a client rather than for the loader's own convenience: keys are flat,
numbers are plain floats, and nothing needs the Python models to interpret.

Pure: takes a :class:`Catalog`, returns a dict.
"""

from __future__ import annotations

from typing import Any

from urdfgen.catalog.loader import Catalog
from urdfgen.catalog.schema import (
    BoxGeometry,
    CylinderGeometry,
    Geometry,
    Link,
    MeshGeometry,
    ModuleDef,
    Origin,
    SphereGeometry,
)

PAYLOAD_VERSION = 1
"""Bumped when the shape changes incompatibly, so a client can refuse to guess."""


def catalog_payload(catalog: Catalog) -> dict[str, Any]:
    """Everything a client needs to list, place and draw the catalog."""
    return {
        "version": PAYLOAD_VERSION,
        "interfaces": [
            {"id": interface.id, "description": interface.description}
            for interface in sorted(catalog.interfaces.values(), key=lambda item: item.id)
        ],
        "materials": {
            material.id: [round(channel, 6) for channel in material.rgba]
            for material in sorted(catalog.materials.values(), key=lambda item: item.id)
        },
        "modules": [module_payload(module) for module in catalog.latest()],
    }


def module_payload(module: ModuleDef) -> dict[str, Any]:
    """One module, with the geometry needed to render it."""
    return {
        "id": module.id,
        "version": module.version,
        "category": module.category,
        "display_name": module.display_name,
        "description": " ".join(module.description.split()),
        "tags": list(module.tags),
        "dof": module.dof,
        "mass": round(module.total_mass, 6),
        "root_link": module.root_link,
        "links": [_link(link) for link in module.links],
        "joints": [
            {
                "name": joint.name,
                "type": joint.type,
                "parent": joint.parent,
                "child": joint.child,
                "origin": _origin(joint.origin),
                "axis": list(joint.axis),
                "limit": (
                    None
                    if joint.limit is None
                    else {
                        "lower": joint.limit.lower,
                        "upper": joint.limit.upper,
                        "effort": joint.limit.effort,
                        "velocity": joint.limit.velocity,
                    }
                ),
                "movable": joint.is_movable,
            }
            for joint in module.joints
        ],
        "ports": [
            {
                "name": port.name,
                "role": port.role,
                "interface": port.interface,
                "link": port.link,
                "origin": _origin(port.origin),
                "description": " ".join(port.description.split()),
            }
            for port in module.ports
        ],
    }


def _link(link: Link) -> dict[str, Any]:
    inertial = link.inertial
    return {
        "name": link.name,
        "material": link.material,
        "mass": None if inertial is None else inertial.mass,
        "visual": None if link.visual is None else _visual(link),
    }


def _visual(link: Link) -> dict[str, Any]:
    assert link.visual is not None
    return {
        "origin": _origin(link.visual.origin),
        "material": link.visual.material or link.material,
        "geometry": _geometry(link.visual.geometry),
    }


def _geometry(geometry: Geometry) -> dict[str, Any]:
    if isinstance(geometry, BoxGeometry):
        return {"shape": "box", "size": list(geometry.size)}
    if isinstance(geometry, CylinderGeometry):
        return {"shape": "cylinder", "radius": geometry.radius, "length": geometry.length}
    if isinstance(geometry, SphereGeometry):
        return {"shape": "sphere", "radius": geometry.radius}
    if isinstance(geometry, MeshGeometry):
        # A client that cannot fetch the mesh still needs to know one exists,
        # rather than silently drawing nothing.
        return {"shape": "mesh", "mesh": geometry.mesh, "scale": list(geometry.scale)}
    msg = f"unsupported geometry {geometry!r}"
    raise TypeError(msg)


def _origin(origin: Origin) -> dict[str, list[float]]:
    return {"xyz": list(origin.xyz), "rpy": list(origin.rpy)}
