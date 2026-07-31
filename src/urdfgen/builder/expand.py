"""Expand an :class:`AssemblySpec` into a flat, namespaced link/joint tree.

This is deterministic and pure: given a catalog and a spec, expansion always
produces the same result. It performs no validation of its own beyond what it
needs to proceed -- assembly-level checking is :mod:`urdfgen.validation`'s job,
and callers are expected to validate first.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from urdfgen.builder import expression
from urdfgen.builder.inertia import InertiaError, resolve_inertial
from urdfgen.builder.transform import (
    compose,
    invert,
    matrix_to_origin,
    mirror_axis_vector,
    mirror_inertia,
    mirror_limits,
    mirror_origin,
    pose_to_matrix,
)
from urdfgen.catalog.loader import Catalog, CatalogError
from urdfgen.catalog.schema import (
    Collision,
    Geometry,
    Inertial,
    Joint,
    JointLimit,
    Link,
    MeshGeometry,
    ModuleDef,
    Origin,
    Port,
    Visual,
)
from urdfgen.ir.assembly import (
    AssemblySpec,
    Connection,
    MirrorAxis,
    ModuleInstance,
    PortRef,
)

Vec3 = tuple[float, float, float]

CONNECTION_JOINT_TYPES: frozenset[str] = frozenset({"fixed", "continuous", "floating", "planar"})
"""Joint types a ``Connection`` may request.

``revolute`` and ``prismatic`` are excluded on purpose: URDF requires limits
for them and a connection has nowhere to get limits from. Articulation belongs
inside a module (spec section 3).
"""


class ExpansionError(Exception):
    """Raised when a spec cannot be expanded at all.

    Expansion is the step after validation, so reaching this usually means the
    caller skipped validation. The message says what is structurally missing.
    """


@dataclass(frozen=True)
class ExpandedLink:
    """One URDF link, already namespaced and with all transforms resolved."""

    name: str
    instance: str
    local_name: str
    module_id: str
    visual: Visual | None
    collision: Collision | None
    inertial: Inertial | None
    material: str | None


@dataclass(frozen=True)
class ExpandedJoint:
    """One URDF joint. ``instance`` is ``None`` for inter-module connections."""

    name: str
    type: str
    parent: str
    child: str
    origin: Origin
    axis: Vec3
    limit: JointLimit | None
    instance: str | None = None
    local_name: str | None = None

    @property
    def is_movable(self) -> bool:
        return self.type not in {"fixed"}


@dataclass(frozen=True)
class ExpandedPort:
    """A port of an instance, with its owning link resolved to a global name."""

    instance: str
    port: str
    link: str
    interface: str
    role: str
    origin: Origin
    connected: bool

    @property
    def ref(self) -> str:
        return f"{self.instance}.{self.port}"


@dataclass(frozen=True)
class ExpandedRobot:
    """The complete flattened robot, ready for XML generation."""

    name: str
    root_link: str
    links: tuple[ExpandedLink, ...]
    joints: tuple[ExpandedJoint, ...]
    ports: tuple[ExpandedPort, ...]
    modules_used: tuple[tuple[str, str], ...]
    name_map: dict[tuple[str, str], str] = field(default_factory=dict)
    """``(instance, local name) -> global name``, for UI reverse lookup."""

    @property
    def movable_joints(self) -> tuple[ExpandedJoint, ...]:
        return tuple(joint for joint in self.joints if joint.is_movable)

    @property
    def total_mass(self) -> float:
        return sum(link.inertial.mass for link in self.links if link.inertial is not None)

    def link(self, name: str) -> ExpandedLink | None:
        return next((link for link in self.links if link.name == name), None)

    def unconnected_ports(self) -> tuple[ExpandedPort, ...]:
        return tuple(port for port in self.ports if not port.connected)


def global_name(instance_id: str, local_name: str) -> str:
    """Namespace a module-local name (spec section 6.1)."""
    return f"{instance_id}_{local_name}"


def expand(spec: AssemblySpec, catalog: Catalog) -> ExpandedRobot:
    """Flatten ``spec`` against ``catalog``."""
    resolved = {
        instance.id: _resolve_instance(instance, catalog) for instance in _unique_instances(spec)
    }

    if spec.root not in resolved:
        msg = f"root instance {spec.root!r} is not among the assembly's modules"
        raise ExpansionError(msg)

    links: list[ExpandedLink] = []
    joints: list[ExpandedJoint] = []
    name_map: dict[tuple[str, str], str] = {}

    for instance in spec.modules:
        module = resolved[instance.id]
        for link in module.links:
            name = global_name(instance.id, link.name)
            name_map[(instance.id, link.name)] = name
            links.append(
                ExpandedLink(
                    name=name,
                    instance=instance.id,
                    local_name=link.name,
                    module_id=module.id,
                    visual=link.visual,
                    collision=link.collision,
                    inertial=_resolve_inertial(link, module.id),
                    material=link.material,
                )
            )
        for joint in module.joints:
            name = global_name(instance.id, joint.name)
            name_map[(instance.id, joint.name)] = name
            joints.append(
                ExpandedJoint(
                    name=name,
                    type=joint.type,
                    parent=global_name(instance.id, joint.parent),
                    child=global_name(instance.id, joint.child),
                    origin=joint.origin,
                    axis=joint.axis,
                    limit=joint.limit,
                    instance=instance.id,
                    local_name=joint.name,
                )
            )

    used_ports: set[tuple[str, str]] = set()
    for index, connection in enumerate(spec.connections):
        joints.append(_expand_connection(resolved, connection, index))
        used_ports.add((connection.parent.instance, connection.parent.port))
        used_ports.add((connection.child.instance, connection.child.port))

    _reject_duplicate_names(links, joints)

    ports = tuple(
        ExpandedPort(
            instance=instance.id,
            port=port.name,
            link=global_name(instance.id, port.link),
            interface=port.interface,
            role=port.role,
            origin=port.origin,
            connected=(instance.id, port.name) in used_ports,
        )
        for instance in spec.modules
        for port in resolved[instance.id].ports
    )

    modules_used = tuple(sorted({(module.id, module.version) for module in resolved.values()}))

    return ExpandedRobot(
        name=spec.name,
        root_link=global_name(spec.root, resolved[spec.root].root_link),
        links=tuple(links),
        joints=tuple(joints),
        ports=ports,
        modules_used=modules_used,
        name_map=name_map,
    )


def expand_module(
    module_id: str,
    catalog: Catalog,
    version: str | None = None,
    instance_id: str | None = None,
) -> ExpandedRobot:
    """Expand a single catalog module as a standalone one-instance assembly."""
    instance = instance_id or module_id
    spec = AssemblySpec(
        name=module_id,
        root=instance,
        modules=[ModuleInstance(id=instance, module=module_id, version=version)],
    )
    return expand(spec, catalog)


# --- instance resolution --------------------------------------------------


def _reject_duplicate_names(links: Sequence[ExpandedLink], joints: Sequence[ExpandedJoint]) -> None:
    """URDF requires globally unique link and joint names.

    Namespacing makes collisions between modules impossible, so a duplicate
    here means a generated connection joint name clashed with a module's own
    joint. That would silently corrupt the tree, so refuse to build it.
    """
    for kind, names in (
        ("link", [link.name for link in links]),
        ("joint", [joint.name for joint in joints]),
    ):
        seen: set[str] = set()
        for name in names:
            if name in seen:
                msg = f"duplicate {kind} name {name!r} after namespacing"
                raise ExpansionError(msg)
            seen.add(name)


def _unique_instances(spec: AssemblySpec) -> list[ModuleInstance]:
    """Instances keyed by id, rejecting duplicates.

    Duplicate ids are also reported by the validation layer, but expansion has
    to refuse them too: silently keeping one would produce a robot the user
    never asked for.
    """
    seen: dict[str, ModuleInstance] = {}
    for instance in spec.modules:
        if instance.id in seen:
            msg = f"duplicate module instance id {instance.id!r}"
            raise ExpansionError(msg)
        seen[instance.id] = instance
    return list(seen.values())


def _resolve_instance(instance: ModuleInstance, catalog: Catalog) -> ModuleDef:
    """Fetch the module and apply this instance's parameters and mirroring."""
    try:
        module = catalog.get(instance.module, instance.version)
    except CatalogError as exc:
        msg = f"instance {instance.id!r}: {exc}"
        raise ExpansionError(msg) from exc

    module = apply_parameters(module, instance.parameters)
    return apply_mirror(module, instance.mirror)


def _resolve_inertial(link: Link, module_id: str) -> Inertial | None:
    try:
        return resolve_inertial(link, module_id)
    except InertiaError as exc:
        raise ExpansionError(str(exc)) from exc


# --- parametric modules ---------------------------------------------------


def apply_parameters(module: ModuleDef, values: Mapping[str, float]) -> ModuleDef:
    """Rebuild ``module`` with every declared parameter's effects applied.

    Effects are applied for *all* declared parameters, using the instance value
    when given and the declared default otherwise. Applying defaults too keeps
    the authored YAML and the expressions from drifting apart: the affected
    fields are always whatever the expressions say.
    """
    known = {parameter.name for parameter in module.parameters}
    unexpected = sorted(set(values) - known)
    if unexpected:
        msg = (
            f"module {module.id!r}: unknown parameters {unexpected} "
            f"(known: {sorted(known) or '(none)'})"
        )
        raise ExpansionError(msg)

    if not module.parameters:
        return module

    data: dict[str, Any] = module.model_dump()
    for parameter in module.parameters:
        value = float(values.get(parameter.name, parameter.default))
        if not parameter.min <= value <= parameter.max:
            msg = (
                f"module {module.id!r}: parameter {parameter.name!r} value {value} is outside "
                f"[{parameter.min}, {parameter.max}]"
            )
            raise ExpansionError(msg)
        for effect in parameter.affects:
            try:
                result = expression.evaluate(effect.expr, {expression.VARIABLE_NAME: value})
            except expression.ExpressionError as exc:
                msg = f"module {module.id!r}, parameter {parameter.name!r}: {exc}"
                raise ExpansionError(msg) from exc
            _assign(data, effect.target, result, module.id)

    return ModuleDef.model_validate(data)


_TARGET_COLLECTIONS: dict[str, str] = {"link": "links", "joint": "joints", "port": "ports"}
_COMPONENT_INDEX: dict[str, int] = {"x": 0, "y": 1, "z": 2}


def _assign(data: dict[str, Any], target: str, value: float, module_id: str) -> None:
    """Write ``value`` into the dumped module at ``target``.

    ``target`` looks like ``port:foot.origin.xyz.z`` or
    ``link:shank.inertial.mass``: a kind, the element's name, then a dotted
    path. Vector components are addressed as ``x`` / ``y`` / ``z``.
    """
    kind, _, path = target.partition(":")
    collection = _TARGET_COLLECTIONS.get(kind)
    if not collection or not path:
        kinds = ", ".join(sorted(_TARGET_COLLECTIONS))
        msg = f"module {module_id!r}: malformed affects target {target!r} (expected one of {kinds})"
        raise ExpansionError(msg)

    element_name, *rest = path.split(".")
    if not rest:
        msg = f"module {module_id!r}: affects target {target!r} does not name a field"
        raise ExpansionError(msg)

    entries: Sequence[dict[str, Any]] = data[collection]
    node: Any = next((entry for entry in entries if entry["name"] == element_name), None)
    if node is None:
        msg = f"module {module_id!r}: affects target {target!r} names an unknown {kind}"
        raise ExpansionError(msg)

    for segment in rest[:-1]:
        node = _descend(node, segment, target, module_id)
    _set_leaf(node, rest[-1], value, target, module_id)


def _descend(node: Any, segment: str, target: str, module_id: str) -> Any:
    if not isinstance(node, dict) or segment not in node or node[segment] is None:
        msg = f"module {module_id!r}: affects target {target!r} has no field {segment!r}"
        raise ExpansionError(msg)
    child = node[segment]
    if isinstance(child, tuple):
        # ``model_dump`` renders Vec3 fields as tuples; make them assignable.
        child = list(child)
        node[segment] = child
    return child


def _set_leaf(node: Any, segment: str, value: float, target: str, module_id: str) -> None:
    if isinstance(node, list):
        index = _COMPONENT_INDEX.get(segment)
        if index is None or index >= len(node):
            msg = (
                f"module {module_id!r}: affects target {target!r} indexes a vector with "
                f"{segment!r}; use x, y or z"
            )
            raise ExpansionError(msg)
        node[index] = value
        return
    if isinstance(node, dict) and segment in node:
        node[segment] = value
        return
    msg = f"module {module_id!r}: affects target {target!r} has no field {segment!r}"
    raise ExpansionError(msg)


# --- mirroring ------------------------------------------------------------


def apply_mirror(module: ModuleDef, axis: MirrorAxis) -> ModuleDef:
    """Reflect a whole module across a coordinate plane (spec section 5.4).

    Mirroring is a last resort -- the validation layer always warns about it,
    because negative mesh scales flip normals in some physics engines.
    """
    if axis == "none":
        return module
    return module.model_copy(
        update={
            "links": tuple(_mirror_link(link, axis) for link in module.links),
            "joints": tuple(_mirror_joint(joint, axis) for joint in module.joints),
            "ports": tuple(_mirror_port(port, axis) for port in module.ports),
        }
    )


def _mirror_link(link: Link, axis: MirrorAxis) -> Link:
    visual = (
        None
        if link.visual is None
        else link.visual.model_copy(
            update={
                "origin": mirror_origin(link.visual.origin, axis),
                "geometry": _mirror_geometry(link.visual.geometry, axis),
            }
        )
    )
    collision = (
        None
        if link.collision is None
        else link.collision.model_copy(
            update={
                "origin": mirror_origin(link.collision.origin, axis),
                "geometry": _mirror_geometry(link.collision.geometry, axis),
            }
        )
    )
    inertial = (
        None
        if link.inertial is None
        else link.inertial.model_copy(
            update={
                "origin": mirror_origin(link.inertial.origin, axis),
                "inertia": (
                    None
                    if link.inertial.inertia is None
                    else mirror_inertia(link.inertial.inertia, axis)
                ),
            }
        )
    )
    return link.model_copy(update={"visual": visual, "collision": collision, "inertial": inertial})


def _mirror_geometry(geometry: Geometry, axis: MirrorAxis) -> Geometry:
    """Only meshes need mirroring; primitives are symmetric about their own origin."""
    if not isinstance(geometry, MeshGeometry) or axis == "none":
        return geometry
    index = {"x": 0, "y": 1}[axis]
    scale = list(geometry.scale)
    scale[index] = -scale[index]
    return geometry.model_copy(update={"scale": (scale[0], scale[1], scale[2])})


def _mirror_joint(joint: Joint, axis: MirrorAxis) -> Joint:
    limit = joint.limit
    if limit is not None:
        lower, upper = mirror_limits(limit.lower, limit.upper, axis)
        limit = limit.model_copy(update={"lower": lower, "upper": upper})
    return joint.model_copy(
        update={
            "origin": mirror_origin(joint.origin, axis),
            "axis": mirror_axis_vector(joint.axis, axis),
            "limit": limit,
        }
    )


def _mirror_port(port: Port, axis: MirrorAxis) -> Port:
    return port.model_copy(update={"origin": mirror_origin(port.origin, axis)})


# --- connections ----------------------------------------------------------


def _expand_connection(
    resolved: Mapping[str, ModuleDef],
    connection: Connection,
    index: int,
) -> ExpandedJoint:
    parent_module = _lookup_module(resolved, connection.parent.instance, index)
    child_module = _lookup_module(resolved, connection.child.instance, index)

    parent_port = _lookup_port(parent_module, connection.parent, index)
    child_port = _lookup_port(child_module, connection.child, index)

    if child_port.link != child_module.root_link:
        msg = (
            f"connection {index}: child port {connection.child} sits on link "
            f"{child_port.link!r}, but only the module's root_link "
            f"({child_module.root_link!r}) can be attached to a parent"
        )
        raise ExpansionError(msg)

    joint_type = connection.joint or "fixed"
    if joint_type not in CONNECTION_JOINT_TYPES:
        allowed = ", ".join(sorted(CONNECTION_JOINT_TYPES))
        msg = (
            f"connection {index}: joint type {joint_type!r} is not usable for a connection "
            f"(allowed: {allowed}); put articulation inside a module instead"
        )
        raise ExpansionError(msg)

    # T = T_parent_port . T_offset . T_child_port^-1  (spec section 6.2)
    transform = compose(
        pose_to_matrix(parent_port.origin),
        pose_to_matrix(connection.offset),
        invert(pose_to_matrix(child_port.origin)),
    )

    return ExpandedJoint(
        name=f"{connection.parent.instance}_{parent_port.name}_to_{connection.child.instance}",
        type=joint_type,
        parent=global_name(connection.parent.instance, parent_port.link),
        child=global_name(connection.child.instance, child_module.root_link),
        origin=matrix_to_origin(transform),
        axis=(1.0, 0.0, 0.0),
        limit=None,
    )


def _lookup_module(resolved: Mapping[str, ModuleDef], instance_id: str, index: int) -> ModuleDef:
    module = resolved.get(instance_id)
    if module is None:
        msg = f"connection {index}: unknown module instance {instance_id!r}"
        raise ExpansionError(msg)
    return module


def _lookup_port(module: ModuleDef, ref: PortRef, index: int) -> Port:
    port = module.port(ref.port)
    if port is None:
        available = ", ".join(p.name for p in module.ports) or "(none)"
        msg = (
            f"connection {index}: module {module.id!r} has no port {ref.port!r} "
            f"(available: {available})"
        )
        raise ExpansionError(msg)
    return port
