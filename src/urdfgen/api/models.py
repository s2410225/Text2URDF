"""Request and response bodies for the HTTP API.

These are transport shapes only. Nothing here decides anything: every field is
either an ``AssemblySpec`` on its way in, or the result of a function in
``validation`` / ``builder`` on its way out. Keeping the wire types separate
from the IR means the API can grow a field without touching what the LLM emits.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from urdfgen.builder.expand import ExpandedRobot
from urdfgen.builder.urdf import DEFAULT_PACKAGE_NAME
from urdfgen.ir.assembly import AssemblySpec
from urdfgen.llm.repair import MAX_ATTEMPTS, AssemblyOutcome
from urdfgen.validation.result import ValidationError


class Wire(BaseModel):
    """Base for API bodies: unknown fields are rejected, as in the IR."""

    model_config = ConfigDict(extra="forbid")


# --- requests -------------------------------------------------------------


class SpecRequest(Wire):
    spec: AssemblySpec


class BuildRequest(SpecRequest):
    package_name: str = Field(default=DEFAULT_PACKAGE_NAME, min_length=1)

    skip_validation: bool = False
    """Build a spec that failed validation, to see what it would have produced.

    The same escape hatch ``urdfgen build --skip-validation`` offers, and the
    same warning applies: the result is not expected to be usable.
    """


class AssembleRequest(Wire):
    text: str = Field(min_length=1)
    max_attempts: int = Field(default=MAX_ATTEMPTS, ge=1, le=5)


# --- responses ------------------------------------------------------------


class Finding(Wire):
    """One :class:`~urdfgen.validation.result.ValidationError`, as JSON."""

    code: str
    severity: Literal["error", "warning"]
    message: str
    path: str = ""

    @classmethod
    def of(cls, finding: ValidationError) -> Finding:
        return cls(
            code=finding.code,
            severity=finding.severity,
            message=finding.message,
            path=finding.path,
        )


class Report(Wire):
    """What the validation layer said, with the counts a UI wants to show."""

    findings: list[Finding]
    errors: int
    warnings: int
    ok: bool
    """No errors. Warnings do not make an assembly unbuildable."""

    @classmethod
    def of(cls, findings: list[ValidationError]) -> Report:
        errors = sum(1 for finding in findings if finding.is_error)
        return cls(
            findings=[Finding.of(finding) for finding in findings],
            errors=errors,
            warnings=len(findings) - errors,
            ok=errors == 0,
        )


class ModuleUse(Wire):
    id: str
    version: str


class Summary(Wire):
    """The figures a client would otherwise have to derive from the XML."""

    name: str
    root_link: str
    links: int
    joints: int
    movable_joints: int
    mass: float
    unconnected_ports: list[str]
    modules_used: list[ModuleUse]

    @classmethod
    def of(cls, robot: ExpandedRobot) -> Summary:
        return cls(
            name=robot.name,
            root_link=robot.root_link,
            links=len(robot.links),
            joints=len(robot.joints),
            movable_joints=len(robot.movable_joints),
            mass=round(robot.total_mass, 6),
            unconnected_ports=[port.ref for port in robot.unconnected_ports()],
            modules_used=[
                ModuleUse(id=module_id, version=version)
                for module_id, version in robot.modules_used
            ],
        )


class BuildResponse(Wire):
    urdf: str
    summary: Summary
    report: Report


class AssembleResponse(Wire):
    spec: AssemblySpec | None
    report: Report
    attempts: int
    succeeded: bool

    schema_enforced: bool
    """False when the server never constrained generation to the schema.

    Worth showing: without it the catalog-id enum was a suggestion rather than
    a constraint, so a plausible-looking result deserves more scepticism.
    """

    @classmethod
    def of(cls, outcome: AssemblyOutcome) -> AssembleResponse:
        return cls(
            spec=outcome.spec,
            report=Report.of(outcome.findings),
            attempts=outcome.attempts,
            succeeded=outcome.succeeded,
            schema_enforced=outcome.schema_enforced,
        )


class LLMStatus(Wire):
    configured: bool
    model: str | None = None
    base_url: str | None = None
    detail: str | None = None
    """Why it is unavailable, when it is."""


class Health(Wire):
    status: Literal["ok"] = "ok"
    payload_version: int
    modules: int
    catalog_root: str | None
    web_app: bool
    """Whether a built frontend is being served from this process."""

    llm: LLMStatus


class DefaultAssembly(Wire):
    spec: AssemblySpec
