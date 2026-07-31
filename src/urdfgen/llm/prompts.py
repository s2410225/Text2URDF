"""Prompts for the two generation stages and the repair loop.

The system prompt states the six rules spec section 4.4 requires, and carries
the catalog digest so the model chooses from what exists. It says nothing about
geometry, because there is nothing for the model to decide there.

Input may be Japanese or English (spec section 4.1); the prompts say so rather
than assuming one.
"""

from __future__ import annotations

import json

from urdfgen.catalog.digest import catalog_digest
from urdfgen.catalog.loader import Catalog
from urdfgen.llm.client import Message
from urdfgen.llm.schema_gen import Schema
from urdfgen.validation.result import ValidationError

RULES = """\
Rules you must follow:

1. Never invent shapes, dimensions, masses or inertia. Every module already
   defines its own geometry. You only choose which modules to use, how many,
   and which ports to join.
2. If the request needs a module that is not in the catalog, do not substitute
   something else. Record the missing module in `notes` and build the rest.
3. Each port may be used at most once.
4. The structure must be a tree: every part has exactly one parent, and there
   are no loops.
5. Left/right asymmetric modules end in `_left` or `_right`. Use the one that
   matches the mounting side, e.g. a `_right` leg on a `leg_mount_fr` port.
6. Name each instance after its role, such as `body`, `leg_fl`, `arm_main`.
"""

LANGUAGE_NOTE = (
    "The request may be written in Japanese or English. Interpret either, and "
    "write `notes` in the same language as the request."
)


def system_prompt(catalog: Catalog, module_ids: list[str] | None = None) -> str:
    """The shared preamble: role, rules and the catalog the model may draw from."""
    return "\n".join(
        [
            "You assemble robots from a fixed catalog of pre-designed modules.",
            "",
            "You do not write URDF or XML. You produce a small JSON description of",
            "which modules to use and how they connect. Deterministic code turns that",
            "into URDF, checks it, and reports any mistakes back to you.",
            "",
            LANGUAGE_NOTE,
            "",
            RULES,
            "",
            catalog_digest(catalog, module_ids),
        ]
    )


def module_selection_messages(
    catalog: Catalog,
    text: str,
    schema: Schema,
    *,
    include_schema: bool,
    previous_errors: list[ValidationError] | None = None,
    previous_attempt: str | None = None,
) -> list[Message]:
    """Stage 1: pick the modules and name the instances."""
    instruction = [
        "Choose the modules for this request. Do not describe connections yet;",
        "that is the next step.",
        "",
        "Request:",
        text,
    ]
    return _messages(
        catalog,
        instruction,
        schema,
        include_schema=include_schema,
        previous_errors=previous_errors,
        previous_attempt=previous_attempt,
    )


def connection_messages(
    catalog: Catalog,
    text: str,
    schema: Schema,
    port_options: list[str],
    *,
    include_schema: bool,
    previous_errors: list[ValidationError] | None = None,
) -> list[Message]:
    """Stage 2: join the already-chosen instances, choosing from real ports."""
    instruction = [
        "You have already chosen the modules. Now connect them.",
        "",
        "These are the only ports that exist. Use each at most once, and use",
        "exactly these strings:",
        *(f"  {option}" for option in port_options),
        "",
        "For each connection, `parent` is the port nearer the root and `child` is",
        "the mounting port of the part being attached.",
        "",
        "Request:",
        text,
    ]
    return _messages(
        catalog,
        instruction,
        schema,
        include_schema=include_schema,
        previous_errors=previous_errors,
        previous_attempt=None,
    )


def format_errors(errors: list[ValidationError]) -> str:
    """Render validation findings as feedback the model can act on.

    Warnings are included but marked as such: they are not reasons to change
    the assembly on their own, and a model told to fix everything will churn.
    """
    lines: list[str] = []
    for finding in errors:
        prefix = "MUST FIX" if finding.is_error else "note"
        location = f" ({finding.path})" if finding.path else ""
        lines.append(f"- [{prefix}]{location} {finding.message}")
    return "\n".join(lines)


def _messages(
    catalog: Catalog,
    instruction: list[str],
    schema: Schema,
    *,
    include_schema: bool,
    previous_errors: list[ValidationError] | None,
    previous_attempt: str | None,
) -> list[Message]:
    body = list(instruction)

    if previous_attempt:
        body += ["", "Your previous answer was:", previous_attempt]

    if previous_errors:
        body += [
            "",
            "That attempt was rejected by the checker. Fix these and answer again:",
            format_errors(previous_errors),
        ]

    if include_schema:
        # Only needed when the server would not enforce the schema itself; when
        # it does, repeating it wastes context that small models cannot spare.
        body += [
            "",
            "Reply with a single JSON object matching this schema, and nothing else:",
            json.dumps(schema, ensure_ascii=False),
        ]

    return [
        {"role": "system", "content": system_prompt(catalog)},
        {"role": "user", "content": "\n".join(body)},
    ]
