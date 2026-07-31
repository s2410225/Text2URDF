"""Pydantic models for ``module.yaml`` and ``interfaces.yaml``.

The validators here only enforce constraints that are *internal to a single
module* (name uniqueness, referential integrity between links/joints/ports).
Cross-module and assembly-level checks belong to :mod:`urdfgen.validation`,
which reports problems as ``ValidationError`` lists instead of raising.

Every model is frozen and forbids unknown fields: module definitions are static
data, and a typo in a hand-written YAML file should fail loudly at load time.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Vec3 = tuple[float, float, float]

JointType = Literal["revolute", "continuous", "prismatic", "fixed", "floating", "planar"]
PortRole = Literal["male", "female"]
Category = Literal["body", "leg", "arm", "gripper", "sensor", "adapter", "wheel"]

MOVABLE_JOINT_TYPES: frozenset[str] = frozenset(
    {"revolute", "continuous", "prismatic", "floating", "planar"}
)
"""Joint types that contribute at least one degree of freedom."""

LIMIT_REQUIRED_JOINT_TYPES: frozenset[str] = frozenset({"revolute", "prismatic"})
"""Joint types for which URDF requires a ``<limit>`` element."""

_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")
_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

Name = Annotated[str, Field(pattern=_NAME_RE.pattern, min_length=1)]


class Frozen(BaseModel):
    """Base for all catalog models: immutable and strict about unknown keys."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class Origin(Frozen):
    """A pose relative to the enclosing frame. ``rpy`` is in radians."""

    xyz: Vec3 = (0.0, 0.0, 0.0)
    rpy: Vec3 = (0.0, 0.0, 0.0)


class Inertia(Frozen):
    """Inertia tensor about the centre of mass, in the ``inertial`` origin frame."""

    ixx: float
    iyy: float
    izz: float
    ixy: float = 0.0
    ixz: float = 0.0
    iyz: float = 0.0


class Inertial(Frozen):
    """Mass properties measured from CAD. Never recomputed from meshes.

    ``inertia`` may be omitted only for primitive-shape links, in which case it
    is computed from the geometry and ``mass`` by the appendix A formulas (spec
    section 2.5). ``mass`` is always declared -- there is nothing to derive it
    from.
    """

    mass: float = Field(gt=0.0)
    origin: Origin = Origin()
    inertia: Inertia | None = None


class BoxGeometry(Frozen):
    shape: Literal["box"] = "box"
    size: Vec3


class CylinderGeometry(Frozen):
    shape: Literal["cylinder"] = "cylinder"
    radius: float = Field(gt=0.0)
    length: float = Field(gt=0.0)


class SphereGeometry(Frozen):
    shape: Literal["sphere"] = "sphere"
    radius: float = Field(gt=0.0)


class MeshGeometry(Frozen):
    shape: Literal["mesh"] = "mesh"
    mesh: str
    scale: Vec3 = (1.0, 1.0, 1.0)


Geometry = Annotated[
    BoxGeometry | CylinderGeometry | SphereGeometry | MeshGeometry,
    Field(discriminator="shape"),
]

PRIMITIVE_SHAPES: frozenset[str] = frozenset({"box", "cylinder", "sphere"})


class Visual(Frozen):
    geometry: Geometry
    origin: Origin = Origin()
    material: str | None = None


class Collision(Frozen):
    geometry: Geometry
    origin: Origin = Origin()


class JointLimit(Frozen):
    lower: float = 0.0
    upper: float = 0.0
    effort: float = Field(gt=0.0)
    velocity: float = Field(gt=0.0)

    @model_validator(mode="after")
    def _ordered(self) -> JointLimit:
        if self.lower > self.upper:
            msg = f"joint limit lower ({self.lower}) exceeds upper ({self.upper})"
            raise ValueError(msg)
        return self


class Link(Frozen):
    name: Name
    visual: Visual | None = None
    collision: Collision | None = None
    inertial: Inertial | None = None
    material: str | None = None

    @property
    def is_primitive_only(self) -> bool:
        """True when no mesh geometry is referenced by this link.

        Only such links may omit ``inertial.inertia`` and fall back to the
        computed values of :mod:`urdfgen.builder.inertia` (spec section 2.5).
        """
        for element in (self.visual, self.collision):
            if element is not None and element.geometry.shape not in PRIMITIVE_SHAPES:
                return False
        return True

    @property
    def geometry_source(self) -> Visual | Collision | None:
        """The geometry the inertia fallback measures, collision taking priority."""
        return self.collision or self.visual

    @model_validator(mode="after")
    def _inertia_required_for_meshes(self) -> Link:
        if self.inertial is not None and self.inertial.inertia is None:
            if not self.is_primitive_only:
                msg = (
                    f"link {self.name!r} uses mesh geometry and therefore must declare "
                    "'inertial.inertia' explicitly; inertia is never estimated from meshes"
                )
                raise ValueError(msg)
            if self.geometry_source is None:
                msg = (
                    f"link {self.name!r} omits 'inertial.inertia' but has no geometry "
                    "to derive it from"
                )
                raise ValueError(msg)
        return self


class Joint(Frozen):
    name: Name
    type: JointType
    parent: Name
    child: Name
    origin: Origin = Origin()
    axis: Vec3 = (1.0, 0.0, 0.0)
    limit: JointLimit | None = None

    @model_validator(mode="after")
    def _check(self) -> Joint:
        if self.parent == self.child:
            msg = f"joint {self.name!r} connects link {self.parent!r} to itself"
            raise ValueError(msg)
        if self.type in LIMIT_REQUIRED_JOINT_TYPES and self.limit is None:
            msg = f"joint {self.name!r} of type {self.type!r} requires a 'limit'"
            raise ValueError(msg)
        if self.type in MOVABLE_JOINT_TYPES and self.axis == (0.0, 0.0, 0.0):
            msg = f"joint {self.name!r} of type {self.type!r} needs a non-zero axis"
            raise ValueError(msg)
        return self

    @property
    def is_movable(self) -> bool:
        return self.type in MOVABLE_JOINT_TYPES


class Port(Frozen):
    """A named attachment frame exposed to other modules."""

    name: Name
    role: PortRole
    interface: str = Field(min_length=1)
    link: Name
    origin: Origin = Origin()
    description: str = ""


class ParameterEffect(Frozen):
    """One assignment applied when a parameter value is bound.

    ``target`` addresses a field of the module, ``expr`` is an arithmetic
    expression whose only free variable is ``value``.
    """

    target: str = Field(min_length=1)
    expr: str = Field(min_length=1)


class Parameter(Frozen):
    name: Name
    unit: str = ""
    default: float
    min: float
    max: float
    affects: tuple[ParameterEffect, ...] = ()

    @model_validator(mode="after")
    def _range(self) -> Parameter:
        if self.min > self.max:
            msg = f"parameter {self.name!r}: min ({self.min}) exceeds max ({self.max})"
            raise ValueError(msg)
        if not self.min <= self.default <= self.max:
            msg = (
                f"parameter {self.name!r}: default ({self.default}) is outside "
                f"[{self.min}, {self.max}]"
            )
            raise ValueError(msg)
        return self


class ModuleDef(Frozen):
    """One catalog entry: a sub-tree of links and joints plus its ports."""

    id: str = Field(min_length=1)
    version: str
    display_name: str = ""
    category: Category
    description: str = ""
    tags: tuple[str, ...] = ()
    root_link: Name
    links: tuple[Link, ...] = Field(min_length=1)
    joints: tuple[Joint, ...] = ()
    ports: tuple[Port, ...] = ()
    parameters: tuple[Parameter, ...] = ()

    @model_validator(mode="after")
    def _check(self) -> ModuleDef:
        if not _SEMVER_RE.match(self.version):
            msg = f"module {self.id!r}: version {self.version!r} is not semver (x.y.z)"
            raise ValueError(msg)

        link_names = _unique(link.name for link in self.links)
        _unique(joint.name for joint in self.joints)
        _unique(port.name for port in self.ports)

        if self.root_link not in link_names:
            msg = f"module {self.id!r}: root_link {self.root_link!r} is not one of its links"
            raise ValueError(msg)

        for joint in self.joints:
            for role, link in (("parent", joint.parent), ("child", joint.child)):
                if link not in link_names:
                    msg = (
                        f"module {self.id!r}: joint {joint.name!r} references unknown "
                        f"{role} link {link!r}"
                    )
                    raise ValueError(msg)

        for port in self.ports:
            if port.link not in link_names:
                msg = (
                    f"module {self.id!r}: port {port.name!r} references unknown link {port.link!r}"
                )
                raise ValueError(msg)

        self._check_internal_tree(link_names)
        return self

    def _check_internal_tree(self, link_names: set[str]) -> None:
        """The module's own links must form a single tree rooted at ``root_link``."""
        parent_of: dict[str, str] = {}
        for joint in self.joints:
            if joint.child in parent_of:
                msg = (
                    f"module {self.id!r}: link {joint.child!r} has more than one parent joint "
                    "(the module must be a tree)"
                )
                raise ValueError(msg)
            parent_of[joint.child] = joint.parent

        if self.root_link in parent_of:
            msg = f"module {self.id!r}: root_link {self.root_link!r} must not have a parent joint"
            raise ValueError(msg)

        reachable = {self.root_link}
        children: dict[str, list[str]] = {}
        for child, parent in parent_of.items():
            children.setdefault(parent, []).append(child)
        stack = [self.root_link]
        while stack:
            for child in children.get(stack.pop(), ()):
                if child not in reachable:
                    reachable.add(child)
                    stack.append(child)

        unreachable = sorted(link_names - reachable)
        if unreachable:
            msg = (
                f"module {self.id!r}: links {unreachable} are not reachable from "
                f"root_link {self.root_link!r}"
            )
            raise ValueError(msg)

    @property
    def dof(self) -> int:
        """Number of movable joints inside this module."""
        return sum(1 for joint in self.joints if joint.is_movable)

    @property
    def total_mass(self) -> float:
        """Sum of the declared link masses. Links without ``inertial`` count as 0."""
        return sum(link.inertial.mass for link in self.links if link.inertial is not None)

    def link(self, name: str) -> Link | None:
        return next((link for link in self.links if link.name == name), None)

    def port(self, name: str) -> Port | None:
        return next((port for port in self.ports if port.name == name), None)

    def parameter(self, name: str) -> Parameter | None:
        return next((param for param in self.parameters if param.name == name), None)


class InterfaceDef(Frozen):
    """A mechanical interface standard. Two ports may join only if these match."""

    id: str = Field(min_length=1)
    description: str = ""


class InterfaceFile(Frozen):
    interfaces: tuple[InterfaceDef, ...] = ()


class MaterialDef(Frozen):
    """A named appearance shared by every module.

    Links reference materials by name only. Keeping the colours in one catalog
    file is what makes deduplicating ``<material>`` elements in the generated
    URDF trivial (spec section 6.3).
    """

    id: str = Field(min_length=1)
    rgba: tuple[float, float, float, float]
    description: str = ""

    @model_validator(mode="after")
    def _range(self) -> MaterialDef:
        if any(not 0.0 <= component <= 1.0 for component in self.rgba):
            msg = f"material {self.id!r}: rgba components must lie in [0, 1]"
            raise ValueError(msg)
        return self


class MaterialFile(Frozen):
    materials: tuple[MaterialDef, ...] = ()


def _unique(names: Iterable[str]) -> set[str]:
    """Collect ``names`` into a set, rejecting duplicates."""
    seen: set[str] = set()
    for name in names:
        if name in seen:
            msg = f"duplicate name {name!r}"
            raise ValueError(msg)
        seen.add(name)
    return seen
