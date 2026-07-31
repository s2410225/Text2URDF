"""Loading of the module catalog from disk.

This is the only module in the package that touches the filesystem. Everything
downstream operates on the in-memory :class:`Catalog`.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError as PydanticValidationError

from urdfgen.catalog.schema import (
    InterfaceDef,
    InterfaceFile,
    MaterialDef,
    MaterialFile,
    ModuleDef,
)

MODULE_FILENAME = "module.yaml"
INTERFACES_FILENAME = "interfaces.yaml"
MATERIALS_FILENAME = "materials.yaml"
CATALOG_INDEX_FILENAME = "catalog.yaml"


class CatalogError(Exception):
    """Raised when the catalog on disk cannot be loaded or is malformed.

    This is a load-time failure of static data, not an assembly validation
    problem -- those are reported as ``ValidationError`` lists instead.
    """


@dataclass(frozen=True)
class Catalog:
    """An immutable, indexed view of the module catalog.

    ``modules`` maps a module id to its versions, keyed by semver string.
    """

    modules: dict[str, dict[str, ModuleDef]] = field(default_factory=dict)
    interfaces: dict[str, InterfaceDef] = field(default_factory=dict)
    materials: dict[str, MaterialDef] = field(default_factory=dict)
    root: Path | None = None

    def ids(self) -> list[str]:
        return sorted(self.modules)

    def versions(self, module_id: str) -> list[str]:
        """Versions of ``module_id``, oldest first. Empty if unknown."""
        return sorted(self.modules.get(module_id, {}), key=_version_key)

    def has(self, module_id: str, version: str | None = None) -> bool:
        versions = self.modules.get(module_id)
        if versions is None:
            return False
        return version is None or version in versions

    def get(self, module_id: str, version: str | None = None) -> ModuleDef:
        """Return a module definition, resolving ``None`` to the latest version.

        Raises :class:`CatalogError` when the id or version is unknown. Callers
        that must not raise should check :meth:`has` first -- the validation
        layer does exactly that.
        """
        versions = self.modules.get(module_id)
        if not versions:
            msg = f"unknown module id {module_id!r}"
            raise CatalogError(msg)
        if version is None:
            return versions[self.versions(module_id)[-1]]
        try:
            return versions[version]
        except KeyError as exc:
            available = ", ".join(self.versions(module_id))
            msg = f"module {module_id!r} has no version {version!r} (available: {available})"
            raise CatalogError(msg) from exc

    def latest(self) -> Iterator[ModuleDef]:
        """Iterate over the newest version of every module, ordered by id."""
        for module_id in self.ids():
            yield self.get(module_id)

    def by_category(self) -> dict[str, list[ModuleDef]]:
        grouped: dict[str, list[ModuleDef]] = {}
        for module in self.latest():
            grouped.setdefault(module.category, []).append(module)
        return grouped


def default_catalog_root() -> Path:
    """The ``modules/`` directory shipped with the repository."""
    return Path(__file__).resolve().parents[3] / "modules"


def load_catalog(root: Path | None = None) -> Catalog:
    """Read every ``module.yaml`` under ``root`` plus ``interfaces.yaml``."""
    root = (root or default_catalog_root()).resolve()
    if not root.is_dir():
        msg = f"catalog root {root} does not exist"
        raise CatalogError(msg)

    interfaces = _load_interfaces(root / INTERFACES_FILENAME)
    materials = _load_materials(root / MATERIALS_FILENAME)

    modules: dict[str, dict[str, ModuleDef]] = {}
    for path in sorted(root.glob(f"*/{MODULE_FILENAME}")):
        module = _load_module(path)
        if module.id != path.parent.name:
            msg = (
                f"{path}: module id {module.id!r} does not match its directory "
                f"name {path.parent.name!r}"
            )
            raise CatalogError(msg)
        versions = modules.setdefault(module.id, {})
        if module.version in versions:
            msg = f"{path}: duplicate definition of {module.id!r} version {module.version}"
            raise CatalogError(msg)
        versions[module.version] = module

    _check_references(modules, interfaces, materials, root)
    return Catalog(modules=modules, interfaces=interfaces, materials=materials, root=root)


@lru_cache(maxsize=8)
def _load_catalog_cached(root: Path) -> Catalog:
    return load_catalog(root)


def load_catalog_cached(root: Path | None = None) -> Catalog:
    """Cached variant of :func:`load_catalog`, keyed by resolved root path."""
    return _load_catalog_cached((root or default_catalog_root()).resolve())


def catalog_index(catalog: Catalog) -> dict[str, Any]:
    """Build the ``catalog.yaml`` index (generated; not committed, see spec 2.2)."""
    return {
        "modules": [
            {
                "id": module.id,
                "versions": catalog.versions(module.id),
                "category": module.category,
                "display_name": module.display_name,
                "dof": module.dof,
                "mass": round(module.total_mass, 6),
                "ports": [
                    {"name": port.name, "role": port.role, "interface": port.interface}
                    for port in module.ports
                ],
            }
            for module in catalog.latest()
        ],
        "interfaces": [
            {"id": interface.id, "description": interface.description}
            for interface in sorted(catalog.interfaces.values(), key=lambda i: i.id)
        ],
    }


def write_catalog_index(catalog: Catalog, path: Path | None = None) -> Path:
    """Write the generated index next to the modules and return its path."""
    if path is None:
        if catalog.root is None:
            msg = "catalog has no root directory; pass an explicit path"
            raise CatalogError(msg)
        path = catalog.root / CATALOG_INDEX_FILENAME
    header = "# Generated by urdfgen from module.yaml files. Do not edit or commit.\n"
    body = yaml.safe_dump(catalog_index(catalog), sort_keys=False, allow_unicode=True)
    path.write_text(header + body, encoding="utf-8")
    return path


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        msg = f"{path}: cannot read YAML: {exc}"
        raise CatalogError(msg) from exc
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        msg = f"{path}: expected a YAML mapping at the top level, got {type(raw).__name__}"
        raise CatalogError(msg)
    return raw


def _load_module(path: Path) -> ModuleDef:
    try:
        return ModuleDef.model_validate(_read_yaml(path))
    except PydanticValidationError as exc:
        msg = f"{path}: invalid module definition:\n{exc}"
        raise CatalogError(msg) from exc


def _load_interfaces(path: Path) -> dict[str, InterfaceDef]:
    if not path.is_file():
        msg = f"missing {path}"
        raise CatalogError(msg)
    try:
        parsed = InterfaceFile.model_validate(_read_yaml(path))
    except PydanticValidationError as exc:
        msg = f"{path}: invalid interface definitions:\n{exc}"
        raise CatalogError(msg) from exc

    by_id: dict[str, InterfaceDef] = {}
    for interface in parsed.interfaces:
        if interface.id in by_id:
            msg = f"{path}: duplicate interface id {interface.id!r}"
            raise CatalogError(msg)
        by_id[interface.id] = interface
    return by_id


def _load_materials(path: Path) -> dict[str, MaterialDef]:
    if not path.is_file():
        return {}
    try:
        parsed = MaterialFile.model_validate(_read_yaml(path))
    except PydanticValidationError as exc:
        msg = f"{path}: invalid material definitions:\n{exc}"
        raise CatalogError(msg) from exc

    by_id: dict[str, MaterialDef] = {}
    for material in parsed.materials:
        if material.id in by_id:
            msg = f"{path}: duplicate material id {material.id!r}"
            raise CatalogError(msg)
        by_id[material.id] = material
    return by_id


def _check_references(
    modules: dict[str, dict[str, ModuleDef]],
    interfaces: dict[str, InterfaceDef],
    materials: dict[str, MaterialDef],
    root: Path,
) -> None:
    for versions in modules.values():
        for module in versions.values():
            source = root / module.id / MODULE_FILENAME
            for port in module.ports:
                if port.interface not in interfaces:
                    msg = (
                        f"{source}: port {port.name!r} uses interface "
                        f"{port.interface!r}, which is not declared in {INTERFACES_FILENAME}"
                    )
                    raise CatalogError(msg)
            for link in module.links:
                names = [link.material, link.visual.material if link.visual else None]
                for name in names:
                    if name is not None and name not in materials:
                        msg = (
                            f"{source}: link {link.name!r} uses material {name!r}, "
                            f"which is not declared in {MATERIALS_FILENAME}"
                        )
                        raise CatalogError(msg)


def _version_key(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))
