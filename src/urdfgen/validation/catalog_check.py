"""Catalog consistency checks (spec section 5.1).

Each function is pure and independent: it takes the spec and the catalog and
returns findings. None of them raise, so a spec that fails one check can still
be run through the others and produce a complete report in one pass.
"""

from __future__ import annotations

from urdfgen.catalog.loader import Catalog
from urdfgen.ir.assembly import AssemblySpec
from urdfgen.validation.result import ValidationError, error


def check_modules_exist(spec: AssemblySpec, catalog: Catalog) -> list[ValidationError]:
    """Every referenced catalog id and pinned version must exist."""
    findings: list[ValidationError] = []
    for index, instance in enumerate(spec.modules):
        path = f"modules[{index}]"
        if not catalog.has(instance.module):
            known = ", ".join(catalog.ids()) or "(catalog is empty)"
            findings.append(
                error(
                    "unknown_module",
                    f"instance {instance.id!r} uses module {instance.module!r}, which is "
                    f"not in the catalog (known: {known})",
                    path,
                )
            )
            continue
        if instance.version is not None and not catalog.has(instance.module, instance.version):
            available = ", ".join(catalog.versions(instance.module))
            findings.append(
                error(
                    "unknown_version",
                    f"instance {instance.id!r} pins {instance.module!r} version "
                    f"{instance.version!r}, which does not exist (available: {available})",
                    path,
                )
            )
    return findings


def check_ports_exist(spec: AssemblySpec, catalog: Catalog) -> list[ValidationError]:
    """Every ``PortRef`` must name an existing instance and one of its ports."""
    findings: list[ValidationError] = []
    for index, connection in enumerate(spec.connections):
        for side, ref in (("parent", connection.parent), ("child", connection.child)):
            path = f"connections[{index}].{side}"
            instance = spec.instance(ref.instance)
            if instance is None:
                findings.append(
                    error(
                        "unknown_instance",
                        f"connection refers to instance {ref.instance!r}, which is not "
                        "declared in this assembly",
                        path,
                    )
                )
                continue
            if not catalog.has(instance.module, instance.version):
                # Already reported by check_modules_exist; nothing to look up.
                continue
            module = catalog.get(instance.module, instance.version)
            if module.port(ref.port) is None:
                available = ", ".join(port.name for port in module.ports) or "(none)"
                findings.append(
                    error(
                        "unknown_port",
                        f"module {module.id!r} has no port {ref.port!r} (available: {available})",
                        path,
                    )
                )
    return findings


def check_parameters(spec: AssemblySpec, catalog: Catalog) -> list[ValidationError]:
    """Parameter names must be declared and values must lie within their range."""
    findings: list[ValidationError] = []
    for index, instance in enumerate(spec.modules):
        if not instance.parameters:
            continue
        path = f"modules[{index}].parameters"
        if not catalog.has(instance.module, instance.version):
            continue
        module = catalog.get(instance.module, instance.version)
        for name, value in sorted(instance.parameters.items()):
            parameter = module.parameter(name)
            if parameter is None:
                declared = ", ".join(p.name for p in module.parameters) or "(none)"
                findings.append(
                    error(
                        "unknown_parameter",
                        f"module {module.id!r} has no parameter {name!r} (declared: {declared})",
                        path,
                    )
                )
                continue
            if not parameter.min <= value <= parameter.max:
                findings.append(
                    error(
                        "parameter_out_of_range",
                        f"parameter {name!r} = {value} is outside "
                        f"[{parameter.min}, {parameter.max}]",
                        path,
                    )
                )
    return findings


CHECKS = (check_modules_exist, check_ports_exist, check_parameters)
