"""The finding type every validation check returns.

Validation failures are data, not exceptions: a check reports what it found and
the caller decides what to do. Nothing in this package raises on an invalid
assembly (spec section 12).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Severity = Literal["error", "warning"]


@dataclass(frozen=True)
class ValidationError:
    """One finding.

    ``code`` is a stable machine-readable identifier, ``path`` points at the
    offending part of the spec (``modules[2]``, ``connections[0].child``) so a
    UI can highlight it.
    """

    code: str
    severity: Severity
    message: str
    path: str = ""

    @property
    def is_error(self) -> bool:
        return self.severity == "error"

    def __str__(self) -> str:
        location = f" at {self.path}" if self.path else ""
        return f"[{self.severity}] {self.code}{location}: {self.message}"


def error(code: str, message: str, path: str = "") -> ValidationError:
    return ValidationError(code=code, severity="error", message=message, path=path)


def warning(code: str, message: str, path: str = "") -> ValidationError:
    return ValidationError(code=code, severity="warning", message=message, path=path)


def has_errors(findings: list[ValidationError]) -> bool:
    return any(finding.is_error for finding in findings)


def errors_only(findings: list[ValidationError]) -> list[ValidationError]:
    return [finding for finding in findings if finding.is_error]


def warnings_only(findings: list[ValidationError]) -> list[ValidationError]:
    return [finding for finding in findings if not finding.is_error]


def summarise(findings: list[ValidationError]) -> str:
    """A one-line count, for CLI output and error feedback to the LLM."""
    errors = len(errors_only(findings))
    return f"{errors} error(s), {len(findings) - errors} warning(s)"
