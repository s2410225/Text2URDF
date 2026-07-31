"""The ``AssemblySpec`` intermediate representation.

This is the only structure the LLM produces, and it is deliberately tiny: it
names modules and connects their ports. Geometry, mass and inertia never appear
here -- they live in the catalog.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Vec3 = tuple[float, float, float]

MirrorAxis = Literal["none", "x", "y"]


class Spec(BaseModel):
    """Base for IR models: immutable, and unknown fields are rejected.

    Rejecting unknown fields matters because this data comes from an LLM: a
    hallucinated field must fail rather than be silently dropped.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")


class Pose(Spec):
    """A pose. ``rpy`` is in radians."""

    xyz: Vec3 = (0.0, 0.0, 0.0)
    rpy: Vec3 = (0.0, 0.0, 0.0)


class ModuleInstance(Spec):
    """One use of a catalog module inside an assembly."""

    id: str = Field(min_length=1)
    """Instance name, e.g. ``leg_fl``. Unique within the assembly."""

    module: str = Field(min_length=1)
    """Catalog id, e.g. ``leg_3dof_left_v1``."""

    version: str | None = None
    """Pinned catalog version. ``None`` resolves to the latest at build time."""

    parameters: dict[str, float] = Field(default_factory=dict)
    mirror: MirrorAxis = "none"


class PortRef(Spec):
    """A reference to one port of one module instance."""

    instance: str = Field(min_length=1)
    port: str = Field(min_length=1)

    def __str__(self) -> str:
        return f"{self.instance}.{self.port}"


class Connection(Spec):
    """A parent port joined to a child port."""

    parent: PortRef
    child: PortRef

    offset: Pose = Pose()
    """Fine adjustment at the mating surface. Normally zero."""

    joint: str | None = None
    """Joint type for the connection; ``None`` generates a fixed joint.

    Articulation belongs inside modules. Setting this is an exception reserved
    for cases a module cannot express.
    """


class AssemblySpec(Spec):
    """A complete robot as module instances plus port connections."""

    name: str = Field(min_length=1)
    root: str = Field(min_length=1)
    """Instance id of the root module."""

    modules: list[ModuleInstance] = Field(min_length=1)
    connections: list[Connection] = Field(default_factory=list)

    notes: str | None = None
    """Assumptions the LLM made, and any modules it found missing."""

    def instance(self, instance_id: str) -> ModuleInstance | None:
        return next((m for m in self.modules if m.id == instance_id), None)
