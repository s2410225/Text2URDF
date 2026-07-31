"""JSON Schemas generated from the catalog at runtime.

The schemas are what make the assembly layer's guarantees mechanical rather
than advisory: ``module`` can only be a catalog id, and a port reference can
only be a port that the chosen modules actually have (spec sections 4.1, 4.3).

Two shapes are deliberate concessions to running locally. The schemas are flat
-- no ``$defs``, no nesting beyond one array of objects -- because small models
degrade sharply on nested schemas. And a port reference is a single enum string
``"body.leg_mount_fl"`` rather than an object, since one enum of legal values
constrains far better than two independent free-text fields. Splitting the
string back into a ``PortRef`` is deterministic work, done in
:mod:`urdfgen.llm.assemble`.

Everything here is a pure function of the catalog.
"""

from __future__ import annotations

from typing import Any

from urdfgen.catalog.loader import Catalog

Schema = dict[str, Any]

PORT_REFERENCE_SEPARATOR = "."

MODULE_SELECTION_SCHEMA_NAME = "module_selection"
CONNECTION_SCHEMA_NAME = "connection_plan"

INSTANCE_ID_PATTERN = r"^[a-z][a-z0-9_]*$"
"""Instance ids become URDF name prefixes, so keep them to lowercase identifiers."""


def module_selection_schema(catalog: Catalog) -> Schema:
    """Stage 1: which modules, how many, and what to call each instance.

    ``module`` is an enum over the catalog, so a model cannot name a part that
    does not exist. It can still ask for a sensible-but-wrong combination,
    which is what the validation layer is for.
    """
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["name", "modules", "notes"],
        "properties": {
            "name": {
                "type": "string",
                "description": "Short snake_case name for the robot, e.g. quadruped_a.",
            },
            "modules": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["id", "module"],
                    "properties": {
                        "id": {
                            "type": "string",
                            "pattern": INSTANCE_ID_PATTERN,
                            "description": (
                                "Unique instance name describing its role, "
                                "e.g. body, leg_fl, arm_main."
                            ),
                        },
                        "module": {
                            "type": "string",
                            "enum": catalog.ids(),
                            "description": "Catalog module id. Only these exist.",
                        },
                        "mirror": {
                            "type": "string",
                            "enum": ["none", "x", "y"],
                            "description": (
                                "Leave as none. Only mirror when no dedicated "
                                "left/right module exists."
                            ),
                        },
                    },
                },
            },
            "notes": {
                "type": "string",
                "description": (
                    "Assumptions you made. If the request needs a module that is "
                    "not in the catalog, say so here instead of substituting "
                    "something else."
                ),
            },
        },
    }


def connection_schema(instance_ids: list[str], port_options: list[str]) -> Schema:
    """Stage 2: how the already-chosen instances join together.

    Both ends of every connection are drawn from ``port_options``, which is
    built from the instances stage 1 picked -- so the model is choosing from a
    list rather than inventing port names.
    """
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["root", "connections"],
        "properties": {
            "root": {
                "type": "string",
                "enum": instance_ids,
                "description": "The instance everything else hangs off, usually the body.",
            },
            "connections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["parent", "child"],
                    "properties": {
                        "parent": {
                            "type": "string",
                            "enum": port_options,
                            "description": "Port on the side closer to the root.",
                        },
                        "child": {
                            "type": "string",
                            "enum": port_options,
                            "description": (
                                "Port on the part being attached. Must be that "
                                "module's mounting port."
                            ),
                        },
                    },
                },
            },
        },
    }


def format_port_reference(instance_id: str, port_name: str) -> str:
    return f"{instance_id}{PORT_REFERENCE_SEPARATOR}{port_name}"


def parse_port_reference(reference: str) -> tuple[str, str]:
    """Split ``"body.leg_mount_fl"`` into its instance and port.

    Raises :class:`ValueError` for anything malformed. A model that ignores the
    enum must fail loudly here rather than have a guess made on its behalf.
    """
    instance_id, separator, port_name = reference.partition(PORT_REFERENCE_SEPARATOR)
    if not separator or not instance_id or not port_name:
        msg = (
            f"port reference {reference!r} is not in 'instance.port' form; "
            "it must be one of the offered options"
        )
        raise ValueError(msg)
    return instance_id, port_name
