"""Connection and graph-structure checks (spec section 5.2)."""

from __future__ import annotations

from collections import Counter

from urdfgen.catalog.loader import Catalog
from urdfgen.catalog.schema import ModuleDef, Port
from urdfgen.ir.assembly import AssemblySpec, PortRef
from urdfgen.validation.result import ValidationError, error

OPPOSITE_ROLE = {"male": "female", "female": "male"}


def check_instance_ids_unique(spec: AssemblySpec, catalog: Catalog) -> list[ValidationError]:
    """Instance ids become the namespace prefix, so duplicates would collide."""
    del catalog
    counts = Counter(instance.id for instance in spec.modules)
    return [
        error(
            "duplicate_instance_id",
            f"instance id {instance_id!r} is used {count} times; ids must be unique",
            "modules",
        )
        for instance_id, count in sorted(counts.items())
        if count > 1
    ]


def check_root_exists(spec: AssemblySpec, catalog: Catalog) -> list[ValidationError]:
    del catalog
    if spec.instance(spec.root) is None:
        declared = ", ".join(instance.id for instance in spec.modules)
        return [
            error(
                "unknown_root",
                f"root {spec.root!r} is not one of the declared instances ({declared})",
                "root",
            )
        ]
    return []


def check_port_usage(spec: AssemblySpec, catalog: Catalog) -> list[ValidationError]:
    """Each port may take part in at most one connection (spec 4.4)."""
    del catalog
    counts = Counter(
        (ref.instance, ref.port)
        for connection in spec.connections
        for ref in (connection.parent, connection.child)
    )
    return [
        error(
            "port_reused",
            f"port {instance_id}.{port} is used by {count} connections; "
            "a port can only be connected once",
            "connections",
        )
        for (instance_id, port), count in sorted(counts.items())
        if count > 1
    ]


def check_interfaces_match(spec: AssemblySpec, catalog: Catalog) -> list[ValidationError]:
    """Both ends must share an interface and form a male/female pair."""
    findings: list[ValidationError] = []
    for index, connection in enumerate(spec.connections):
        path = f"connections[{index}]"
        parent = _resolve_port(spec, catalog, connection.parent)
        child = _resolve_port(spec, catalog, connection.child)
        if parent is None or child is None:
            # Reported by the catalog checks; there is nothing to compare.
            continue

        if parent.interface != child.interface:
            findings.append(
                error(
                    "interface_mismatch",
                    f"{connection.parent} is {parent.interface} but {connection.child} is "
                    f"{child.interface}; connected ports must share an interface",
                    path,
                )
            )
        if OPPOSITE_ROLE[parent.role] != child.role:
            findings.append(
                error(
                    "role_mismatch",
                    f"{connection.parent} and {connection.child} are both {parent.role}; "
                    "a connection needs one male and one female port",
                    path,
                )
            )
    return findings


def check_child_uses_root_link(spec: AssemblySpec, catalog: Catalog) -> list[ValidationError]:
    """A child must attach through a port that sits on its own ``root_link``.

    Attaching through any other port would make the module's internal joint
    tree flow backwards from the attachment point, which URDF cannot express.
    """
    findings: list[ValidationError] = []
    for index, connection in enumerate(spec.connections):
        module = _resolve_module(spec, catalog, connection.child.instance)
        port = _resolve_port(spec, catalog, connection.child)
        if module is None or port is None:
            continue
        if port.link != module.root_link:
            findings.append(
                error(
                    "child_port_not_on_root_link",
                    f"{connection.child} sits on link {port.link!r}, but a child must be "
                    f"attached through a port on its root_link ({module.root_link!r})",
                    f"connections[{index}].child",
                )
            )
    return findings


def check_tree_structure(spec: AssemblySpec, catalog: Catalog) -> list[ValidationError]:
    """The assembly graph must be a tree rooted at ``spec.root``.

    Three distinct failures are separated here because they need different
    fixes: a module with two parents, a cycle, and a disconnected module.
    """
    del catalog
    findings: list[ValidationError] = []

    parent_of: dict[str, str] = {}
    children: dict[str, list[str]] = {}
    for index, connection in enumerate(spec.connections):
        child = connection.child.instance
        parent = connection.parent.instance
        if child in parent_of:
            findings.append(
                error(
                    "multiple_parents",
                    f"instance {child!r} is attached to both {parent_of[child]!r} and "
                    f"{parent!r}; the assembly must be a tree",
                    f"connections[{index}]",
                )
            )
            continue
        if child == parent:
            findings.append(
                error(
                    "self_connection",
                    f"instance {child!r} is connected to itself",
                    f"connections[{index}]",
                )
            )
            continue
        parent_of[child] = parent
        children.setdefault(parent, []).append(child)

    declared = {instance.id for instance in spec.modules}
    if spec.root not in declared:
        # check_root_exists already reported this; reachability is meaningless.
        return findings

    if spec.root in parent_of:
        findings.append(
            error(
                "root_has_parent",
                f"root instance {spec.root!r} is attached to {parent_of[spec.root]!r}; "
                "the root must not be a child of anything",
                "root",
            )
        )

    reachable = {spec.root}
    stack = [spec.root]
    while stack:
        for child in children.get(stack.pop(), ()):
            if child not in reachable:
                reachable.add(child)
                stack.append(child)

    for orphan in sorted(declared - reachable):
        if orphan in parent_of:
            findings.append(
                error(
                    "cycle_detected",
                    f"instance {orphan!r} is connected but not reachable from the root "
                    f"{spec.root!r}; the connections form a cycle",
                    "connections",
                )
            )
        else:
            findings.append(
                error(
                    "unreachable_instance",
                    f"instance {orphan!r} is not connected to the root {spec.root!r}",
                    "connections",
                )
            )
    return findings


def check_connection_joint_type(spec: AssemblySpec, catalog: Catalog) -> list[ValidationError]:
    """A connection may only request a joint type that needs no limits."""
    del catalog
    from urdfgen.builder.expand import CONNECTION_JOINT_TYPES

    return [
        error(
            "unsupported_connection_joint",
            f"connection joint type {connection.joint!r} is not supported "
            f"(allowed: {', '.join(sorted(CONNECTION_JOINT_TYPES))}); articulation "
            "belongs inside a module",
            f"connections[{index}].joint",
        )
        for index, connection in enumerate(spec.connections)
        if connection.joint is not None and connection.joint not in CONNECTION_JOINT_TYPES
    ]


# --- helpers --------------------------------------------------------------


def _resolve_module(spec: AssemblySpec, catalog: Catalog, instance_id: str) -> ModuleDef | None:
    instance = spec.instance(instance_id)
    if instance is None or not catalog.has(instance.module, instance.version):
        return None
    return catalog.get(instance.module, instance.version)


def _resolve_port(spec: AssemblySpec, catalog: Catalog, ref: PortRef) -> Port | None:
    module = _resolve_module(spec, catalog, ref.instance)
    return None if module is None else module.port(ref.port)


CHECKS = (
    check_instance_ids_unique,
    check_root_exists,
    check_port_usage,
    check_interfaces_match,
    check_child_uses_root_link,
    check_tree_structure,
    check_connection_joint_type,
)
