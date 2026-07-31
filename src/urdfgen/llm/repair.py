"""The repair loop: generate, validate, feed the errors back (spec section 4.5).

A local model will get an assembly wrong sometimes, and the validation layer
already knows exactly why. Handing those messages back is far cheaper than
asking for a better model.

After three attempts the loop gives up, but it does not swallow the failure:
the partial spec and the outstanding findings come back together, so a person
can fix the last mile by hand.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from urdfgen.catalog.loader import Catalog
from urdfgen.ir.assembly import AssemblySpec
from urdfgen.llm.assemble import AssemblyError, AssemblyTrace, assemble
from urdfgen.llm.client import LocalLLMClient
from urdfgen.validation import validate
from urdfgen.validation.result import ValidationError, error, has_errors

MAX_ATTEMPTS = 3


@dataclass
class AssemblyOutcome:
    """The result of a run, successful or not.

    ``spec`` is ``None`` only when no attempt produced something well-formed
    enough to be an ``AssemblySpec`` at all. Whenever there is a spec, it is
    returned even if it failed validation -- an editable starting point beats
    an error message.
    """

    spec: AssemblySpec | None
    findings: list[ValidationError] = field(default_factory=list)
    attempts: int = 0
    traces: list[AssemblyTrace] = field(default_factory=list)

    @property
    def succeeded(self) -> bool:
        return self.spec is not None and not has_errors(self.findings)

    @property
    def schema_enforced(self) -> bool:
        """False if the server never constrained generation to the schema.

        Worth surfacing: without it the catalog-id enum was only a suggestion,
        so a plausible-looking result deserves more scepticism.
        """
        return all(trace.schema_enforced for trace in self.traces)


def assemble_with_repair(
    text: str,
    catalog: Catalog,
    client: LocalLLMClient,
    max_attempts: int = MAX_ATTEMPTS,
) -> AssemblyOutcome:
    """Generate an assembly for ``text``, retrying on validation errors."""
    outcome = AssemblyOutcome(spec=None)
    previous_errors: list[ValidationError] | None = None

    for attempt in range(1, max_attempts + 1):
        outcome.attempts = attempt
        try:
            spec, trace = assemble(text, catalog, client, previous_errors)
        except AssemblyError as exc:
            # The answer was not even shaped like an assembly. Report it in the
            # same vocabulary as a validation finding so the next prompt reads
            # the same way regardless of where the problem came from.
            previous_errors = [error("malformed_assembly", str(exc), "")]
            outcome.findings = previous_errors
            continue

        outcome.traces.append(trace)
        outcome.spec = spec
        outcome.findings = validate(spec, catalog)

        if not has_errors(outcome.findings):
            return outcome

        previous_errors = outcome.findings

    return outcome
