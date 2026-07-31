"""Two-stage generation of an :class:`AssemblySpec` (spec section 4.3).

Asking for modules and connections in one call makes models invent port names,
because at the moment they write a port they have not yet committed to a set of
modules. Splitting the request fixes that: stage 2 is given the exact list of
ports the stage 1 choices produced, and can only pick from it.

Nothing here trusts the model. Every field it returns is re-checked before it
becomes part of a spec, and anything unusable raises :class:`AssemblyError`
rather than being guessed at.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from urdfgen.catalog.digest import instance_port_options
from urdfgen.catalog.loader import Catalog
from urdfgen.ir.assembly import AssemblySpec, Connection, MirrorAxis, ModuleInstance, PortRef
from urdfgen.llm import prompts
from urdfgen.llm.client import Completion, LocalLLMClient, Message
from urdfgen.llm.schema_gen import (
    CONNECTION_SCHEMA_NAME,
    MODULE_SELECTION_SCHEMA_NAME,
    connection_schema,
    module_selection_schema,
    parse_port_reference,
)
from urdfgen.validation.result import ValidationError

MIRROR_VALUES: tuple[MirrorAxis, ...] = ("none", "x", "y")


class AssemblyError(Exception):
    """Raised when the model's answer cannot be turned into an AssemblySpec.

    This is a failure of one attempt, not of the run: the repair loop catches
    it, feeds the reason back to the model, and tries again.
    """


@dataclass
class AssemblyTrace:
    """What the two stages actually exchanged, for diagnostics and tests."""

    module_selection: dict[str, Any] = field(default_factory=dict)
    connection_plan: dict[str, Any] = field(default_factory=dict)
    schema_enforced: bool = True
    """False if the server declined schema-constrained generation at any stage."""


def assemble(
    text: str,
    catalog: Catalog,
    client: LocalLLMClient,
    previous_errors: list[ValidationError] | None = None,
) -> tuple[AssemblySpec, AssemblyTrace]:
    """Turn ``text`` into an ``AssemblySpec`` using two constrained calls."""
    trace = AssemblyTrace()

    selection = _select_modules(text, catalog, client, previous_errors, trace)
    modules = _read_modules(selection, catalog)
    instance_ids = [module.id for module in modules]

    port_options = instance_port_options(catalog, {module.id: module.module for module in modules})
    if not port_options:
        msg = (
            "the chosen modules expose no ports at all, so nothing can be "
            f"connected: {', '.join(instance_ids)}"
        )
        raise AssemblyError(msg)

    plan = _plan_connections(
        text, catalog, client, instance_ids, port_options, previous_errors, trace
    )
    root = _read_root(plan, instance_ids)
    connections = _read_connections(plan, port_options)

    return (
        _build_spec(selection, root, modules, connections),
        trace,
    )


# --- stage 1 --------------------------------------------------------------


def _select_modules(
    text: str,
    catalog: Catalog,
    client: LocalLLMClient,
    previous_errors: list[ValidationError] | None,
    trace: AssemblyTrace,
) -> dict[str, Any]:
    schema = module_selection_schema(catalog)
    completion = _call(
        client,
        lambda include_schema: prompts.module_selection_messages(
            catalog, text, schema, include_schema=include_schema, previous_errors=previous_errors
        ),
        schema,
        MODULE_SELECTION_SCHEMA_NAME,
        trace,
    )
    trace.module_selection = completion.data
    return completion.data


def _read_modules(selection: dict[str, Any], catalog: Catalog) -> list[ModuleInstance]:
    raw = selection.get("modules")
    if not isinstance(raw, list) or not raw:
        msg = "the model chose no modules"
        raise AssemblyError(msg)

    known = set(catalog.ids())
    modules: list[ModuleInstance] = []
    seen: set[str] = set()

    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            msg = f"modules[{index}] is not an object: {entry!r}"
            raise AssemblyError(msg)

        instance_id = _text(entry, "id", f"modules[{index}]")
        module_id = _text(entry, "module", f"modules[{index}]")

        if module_id not in known:
            # A schema enum should have prevented this; a small local model can
            # still ignore it, and silently dropping the entry would produce a
            # robot the user did not ask for.
            msg = (
                f"modules[{index}] names {module_id!r}, which is not in the catalog "
                f"(available: {', '.join(sorted(known))})"
            )
            raise AssemblyError(msg)

        if instance_id in seen:
            msg = f"instance id {instance_id!r} is used more than once"
            raise AssemblyError(msg)
        seen.add(instance_id)

        mirror = str(entry.get("mirror") or "none").strip().lower()
        if mirror not in MIRROR_VALUES:
            msg = f"modules[{index}] has mirror={mirror!r}; expected one of none, x, y"
            raise AssemblyError(msg)

        modules.append(
            ModuleInstance(
                id=instance_id,
                module=module_id,
                # Pin the version that was actually offered, so re-running the
                # same spec later cannot silently pick up a different module
                # (spec section 2.5).
                version=catalog.get(module_id).version,
                mirror=mirror,
            )
        )
    return modules


# --- stage 2 --------------------------------------------------------------


def _plan_connections(
    text: str,
    catalog: Catalog,
    client: LocalLLMClient,
    instance_ids: list[str],
    port_options: list[str],
    previous_errors: list[ValidationError] | None,
    trace: AssemblyTrace,
) -> dict[str, Any]:
    schema = connection_schema(instance_ids, port_options)
    completion = _call(
        client,
        lambda include_schema: prompts.connection_messages(
            catalog,
            text,
            schema,
            port_options,
            include_schema=include_schema,
            previous_errors=previous_errors,
        ),
        schema,
        CONNECTION_SCHEMA_NAME,
        trace,
    )
    trace.connection_plan = completion.data
    return completion.data


def _read_root(plan: dict[str, Any], instance_ids: list[str]) -> str:
    root = str(plan.get("root") or "").strip()
    if root not in instance_ids:
        msg = f"root {root!r} is not one of the chosen instances ({', '.join(instance_ids)})"
        raise AssemblyError(msg)
    return root


def _read_connections(plan: dict[str, Any], port_options: list[str]) -> list[Connection]:
    raw = plan.get("connections", [])
    if not isinstance(raw, list):
        msg = f"connections is not a list: {raw!r}"
        raise AssemblyError(msg)

    allowed = set(port_options)
    connections: list[Connection] = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            msg = f"connections[{index}] is not an object: {entry!r}"
            raise AssemblyError(msg)
        parent = _port(entry, "parent", index, allowed)
        child = _port(entry, "child", index, allowed)
        if parent == child:
            msg = f"connections[{index}] joins {parent} to itself"
            raise AssemblyError(msg)
        connections.append(parent_child(parent, child))
    return connections


def parent_child(parent: PortRef, child: PortRef) -> Connection:
    """A plain fixed connection.

    Articulation lives inside modules, so the assembly layer never asks the
    model for a joint type (spec section 3).
    """
    return Connection(parent=parent, child=child)


def _port(entry: dict[str, Any], key: str, index: int, allowed: set[str]) -> PortRef:
    reference = _text(entry, key, f"connections[{index}]")
    if reference not in allowed:
        msg = f"connections[{index}].{key} is {reference!r}, which was not one of the offered ports"
        raise AssemblyError(msg)
    try:
        instance_id, port_name = parse_port_reference(reference)
    except ValueError as exc:
        raise AssemblyError(f"connections[{index}].{key}: {exc}") from exc
    return PortRef(instance=instance_id, port=port_name)


# --- assembly -------------------------------------------------------------


def _build_spec(
    selection: dict[str, Any],
    root: str,
    modules: list[ModuleInstance],
    connections: list[Connection],
) -> AssemblySpec:
    name = str(selection.get("name") or "").strip() or "robot"
    notes = str(selection.get("notes") or "").strip() or None
    return AssemblySpec(
        name=_slug(name),
        root=root,
        modules=modules,
        connections=connections,
        notes=notes,
    )


def _slug(name: str) -> str:
    """Make a robot name safe to use as a URDF ``robot name`` attribute."""
    cleaned = "".join(char if char.isalnum() or char in "_-" else "_" for char in name)
    return cleaned.strip("_") or "robot"


# --- shared helpers -------------------------------------------------------


def _call(
    client: LocalLLMClient,
    build_messages: Callable[[bool], list[Message]],
    schema: dict[str, Any],
    schema_name: str,
    trace: AssemblyTrace,
) -> Completion:
    """Run one stage, putting the schema in the prompt only when it has to be.

    When the server constrains generation, repeating the schema in the prompt
    just consumes context a small model needs elsewhere. When it does not, the
    prompt is the only place the model can learn the expected shape, so the
    client is handed both versions and picks.
    """
    already_refused = not client.schema_supported
    completion = client.complete_json(
        build_messages(already_refused),
        schema,
        schema_name,
        fallback_messages=None if already_refused else build_messages(True),
    )
    if not completion.used_schema:
        trace.schema_enforced = False
    return completion


def _text(entry: dict[str, Any], key: str, where: str) -> str:
    value = entry.get(key)
    if not isinstance(value, str) or not value.strip():
        msg = f"{where}.{key} is missing or not a string: {json.dumps(entry)[:200]}"
        raise AssemblyError(msg)
    return value.strip()
