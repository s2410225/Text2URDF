"""The catalog summary shown to the LLM (spec section 4.2).

The digest deliberately omits mesh paths, inertia tensors and joint origins.
The LLM only decides *which* module, *how many* and *which port*, so anything
beyond that is noise that invites it to invent geometry.
"""

from __future__ import annotations

from urdfgen.catalog.loader import Catalog
from urdfgen.catalog.schema import ModuleDef

CATEGORY_ORDER: tuple[str, ...] = (
    "body",
    "leg",
    "arm",
    "gripper",
    "wheel",
    "sensor",
    "adapter",
)


def catalog_digest(catalog: Catalog, module_ids: list[str] | None = None) -> str:
    """Render the whole catalog, or just ``module_ids``, as a Markdown summary."""
    modules = [
        module for module in catalog.latest() if module_ids is None or module.id in set(module_ids)
    ]
    lines = ["## Available modules", ""]
    for module in sorted(modules, key=_sort_key):
        lines.extend(module_digest(module))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def module_digest(module: ModuleDef) -> list[str]:
    """One module's entry: identity, a one-line summary, and its ports."""
    lines = [f"### {module.id} (category: {module.category})"]

    headline = module.display_name or module.id
    lines.append(f"{headline}. {module.dof} DOF, {module.total_mass:.2f} kg.")

    description = " ".join(module.description.split())
    if description:
        lines.append(description)

    if module.tags:
        lines.append(f"tags: {', '.join(module.tags)}")

    if module.parameters:
        lines.append("parameters:")
        lines.extend(
            f"  - {parameter.name} ({parameter.unit or 'unitless'}): "
            f"default {parameter.default}, range {parameter.min}..{parameter.max}"
            for parameter in module.parameters
        )

    if module.ports:
        lines.append("ports:")
        lines.extend(
            f"  - {port.name} [{port.role}/{port.interface}]"
            + (f" {' '.join(port.description.split())}" if port.description else "")
            for port in module.ports
        )
    else:
        lines.append("ports: (none)")

    return lines


def instance_port_options(catalog: Catalog, instances: dict[str, str]) -> list[str]:
    """Every ``instance.port`` reference available across ``instances``.

    ``instances`` maps an instance id to a catalog module id. This is the enum
    the second generation stage constrains port references to (spec 4.3), so
    the model cannot name a port that does not exist.
    """
    options: list[str] = []
    for instance_id in sorted(instances):
        module_id = instances[instance_id]
        if not catalog.has(module_id):
            continue
        options.extend(f"{instance_id}.{port.name}" for port in catalog.get(module_id).ports)
    return options


def _sort_key(module: ModuleDef) -> tuple[int, str]:
    try:
        rank = CATEGORY_ORDER.index(module.category)
    except ValueError:
        rank = len(CATEGORY_ORDER)
    return (rank, module.id)
