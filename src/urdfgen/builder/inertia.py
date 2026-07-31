"""Inertia tensors for primitive shapes (spec appendix A).

This is a *fallback*, permitted only for modules whose links use nothing but
``box`` / ``cylinder`` / ``sphere`` geometry and that omit ``inertial``. Module
definitions with declared inertia are never recomputed, and inertia is never
estimated from a mesh (spec section 2.5 and 12).
"""

from __future__ import annotations

from urdfgen.builder.transform import rotate_inertia, rpy_to_matrix
from urdfgen.catalog.schema import (
    BoxGeometry,
    CylinderGeometry,
    Geometry,
    Inertia,
    Inertial,
    Link,
    Origin,
    SphereGeometry,
)


class InertiaError(Exception):
    """Raised when inertia cannot legitimately be computed for a link."""


def box_inertia(mass: float, size: tuple[float, float, float]) -> Inertia:
    """Cuboid about its centre, axis-aligned."""
    x, y, z = size
    return Inertia(
        ixx=mass * (y * y + z * z) / 12.0,
        iyy=mass * (x * x + z * z) / 12.0,
        izz=mass * (x * x + y * y) / 12.0,
    )


def cylinder_inertia(mass: float, radius: float, length: float) -> Inertia:
    """Solid cylinder about its centre, symmetry axis along Z."""
    radial = mass * (3.0 * radius * radius + length * length) / 12.0
    return Inertia(ixx=radial, iyy=radial, izz=mass * radius * radius / 2.0)


def sphere_inertia(mass: float, radius: float) -> Inertia:
    """Solid sphere about its centre."""
    value = 2.0 * mass * radius * radius / 5.0
    return Inertia(ixx=value, iyy=value, izz=value)


def primitive_inertia(mass: float, geometry: Geometry) -> Inertia:
    """Principal-axis inertia of a primitive, about its own centre.

    Raises :class:`InertiaError` for mesh geometry, which must always declare
    its inertia in the module definition.
    """
    if mass <= 0.0:
        msg = f"mass must be positive, got {mass}"
        raise InertiaError(msg)
    if isinstance(geometry, BoxGeometry):
        return box_inertia(mass, geometry.size)
    if isinstance(geometry, CylinderGeometry):
        return cylinder_inertia(mass, geometry.radius, geometry.length)
    if isinstance(geometry, SphereGeometry):
        return sphere_inertia(mass, geometry.radius)
    msg = (
        f"cannot compute inertia for {geometry.shape!r} geometry; mesh-based links "
        "must declare 'inertial' in their module definition"
    )
    raise InertiaError(msg)


def primitive_inertial(mass: float, geometry: Geometry, origin: Origin) -> Inertial:
    """Build a full ``inertial`` block for a primitive placed at ``origin``.

    The tensor is expressed about the centre of mass. When ``origin.rpy`` is
    non-zero the principal-axis tensor is rotated with ``I' = R I R^T``, which
    generally produces non-zero off-diagonal terms.
    """
    tensor = primitive_inertia(mass, geometry)
    if origin.rpy != (0.0, 0.0, 0.0):
        tensor = rotate_inertia(tensor, rpy_to_matrix(origin.rpy))
    return Inertial(mass=mass, origin=origin, inertia=tensor)


def resolve_inertial(link: Link, module_id: str) -> Inertial | None:
    """Return the link's mass properties, filling in ``inertia`` only if allowed.

    A declared tensor is returned untouched -- catalog inertia is never
    overwritten by computation. ``None`` means the link declares no mass at
    all; the validation layer reports that, rather than this function guessing
    a value.
    """
    inertial = link.inertial
    if inertial is None or inertial.inertia is not None:
        return inertial

    if not link.is_primitive_only:
        msg = (
            f"module {module_id!r}, link {link.name!r}: mesh geometry requires an "
            "explicit 'inertial.inertia' block"
        )
        raise InertiaError(msg)

    source = link.geometry_source
    if source is None:
        msg = (
            f"module {module_id!r}, link {link.name!r}: no geometry to derive inertia from; "
            "declare 'inertial.inertia' explicitly"
        )
        raise InertiaError(msg)

    # The declared inertial origin locates the centre of mass; the geometry's
    # own origin orients the primitive. Appendix A gives the tensor about the
    # centre of mass, so only the orientation matters here.
    tensor = primitive_inertia(inertial.mass, source.geometry)
    if source.origin.rpy != (0.0, 0.0, 0.0):
        tensor = rotate_inertia(tensor, rpy_to_matrix(source.origin.rpy))
    return Inertial(mass=inertial.mass, origin=inertial.origin, inertia=tensor)
