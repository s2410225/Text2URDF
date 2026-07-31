"""Command line entry point.

``list`` / ``show`` / ``digest`` / ``index`` inspect the catalog, ``build`` and
``validate`` work on ``AssemblySpec`` documents, and ``assemble`` / ``generate``
start from natural language and need a local LLM (see ``llm/config.py`` for the
environment variables).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError as PydanticValidationError

from urdfgen.builder.expand import ExpandedRobot, ExpansionError, expand, expand_module
from urdfgen.builder.urdf import DEFAULT_PACKAGE_NAME, build_urdf_string
from urdfgen.catalog.loader import Catalog, CatalogError, load_catalog, write_catalog_index
from urdfgen.ir.assembly import AssemblySpec
from urdfgen.llm.client import LLMError, LocalLLMClient
from urdfgen.llm.config import LLMConfigError, load_config
from urdfgen.llm.repair import MAX_ATTEMPTS, AssemblyOutcome, assemble_with_repair
from urdfgen.preview import VIEWS, PreviewError, render_file
from urdfgen.preview.svg import DEFAULT_WIDTH
from urdfgen.validation import validate
from urdfgen.validation.result import ValidationError, summarise

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        catalog = load_catalog(args.catalog)
    except CatalogError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_FAILED

    handler = args.handler
    assert callable(handler)
    result: int = handler(args, catalog)
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="urdfgen", description=__doc__)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=None,
        metavar="DIR",
        help="module catalog directory (default: the repository's modules/)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    listing = subparsers.add_parser("list", help="list catalog modules")
    listing.add_argument("--category", default=None, help="show only this category")
    listing.set_defaults(handler=_cmd_list)

    show = subparsers.add_parser("show", help="show one module in detail")
    show.add_argument("module_id")
    show.add_argument("--version", default=None)
    show.set_defaults(handler=_cmd_show)

    index = subparsers.add_parser("index", help="regenerate modules/catalog.yaml")
    index.set_defaults(handler=_cmd_index)

    digest = subparsers.add_parser("digest", help="print the LLM-facing catalog summary")
    digest.set_defaults(handler=_cmd_digest)

    build_module = subparsers.add_parser(
        "build-module", help="build a single module into a standalone URDF"
    )
    build_module.add_argument("module_id")
    build_module.add_argument("--version", default=None)
    _add_build_options(build_module)
    build_module.set_defaults(handler=_cmd_build_module)

    build = subparsers.add_parser("build", help="build an AssemblySpec into a URDF")
    build.add_argument("spec", type=Path, help="path to an AssemblySpec JSON file")
    build.add_argument(
        "--skip-validation",
        action="store_true",
        help="build even if validation reports errors (produces invalid URDF)",
    )
    _add_build_options(build)
    build.set_defaults(handler=_cmd_build)

    check = subparsers.add_parser("validate", help="validate an AssemblySpec")
    check.add_argument("spec", type=Path, help="path to an AssemblySpec JSON file")
    check.set_defaults(handler=_cmd_validate)

    assemble = subparsers.add_parser(
        "assemble", help="turn a natural-language request into an AssemblySpec"
    )
    assemble.add_argument("text", help="what to build, in Japanese or English")
    assemble.add_argument(
        "-o", "--output", type=Path, default=None, help="output file (default: stdout)"
    )
    _add_llm_options(assemble)
    assemble.set_defaults(handler=_cmd_assemble)

    generate = subparsers.add_parser(
        "generate", help="turn a natural-language request straight into a URDF"
    )
    generate.add_argument("text", help="what to build, in Japanese or English")
    _add_build_options(generate)
    _add_llm_options(generate)
    generate.set_defaults(handler=_cmd_generate)

    preview = subparsers.add_parser("preview", help="draw a URDF as an SVG")
    preview.add_argument("urdf", type=Path, help="path to a URDF file")
    preview.add_argument(
        "-o", "--output", type=Path, default=None, help="output file (default: stdout)"
    )
    preview.add_argument(
        "--joint",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="joint position, in radians or metres; repeatable (default: all zero)",
    )
    preview.add_argument(
        "--view", default="iso", choices=sorted(VIEWS), help="camera direction (default: iso)"
    )
    preview.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="image width in pixels")
    preview.add_argument("--title", default=None, help="caption (default: the robot's name)")
    preview.add_argument("--no-grid", action="store_true", help="omit the ground grid")
    preview.add_argument("--no-axes", action="store_true", help="omit the origin axes")
    preview.set_defaults(handler=_cmd_preview)

    return parser


def _add_build_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-o", "--output", type=Path, default=None, help="output file (default: stdout)"
    )
    parser.add_argument(
        "--package", default=DEFAULT_PACKAGE_NAME, help="ROS package name used in mesh URIs"
    )


def _add_llm_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=None, help="overrides LLM_MODEL")
    parser.add_argument("--base-url", default=None, help="overrides LLM_BASE_URL")
    parser.add_argument(
        "--attempts",
        type=int,
        default=MAX_ATTEMPTS,
        help=f"how many times to retry after validation errors (default: {MAX_ATTEMPTS})",
    )


# --- commands -------------------------------------------------------------


def _cmd_list(args: argparse.Namespace, catalog: Catalog) -> int:
    grouped = catalog.by_category()
    for category in sorted(grouped):
        if args.category and category != args.category:
            continue
        print(f"{category}:")
        for module in grouped[category]:
            ports = ", ".join(port.name for port in module.ports) or "(no ports)"
            print(
                f"  {module.id} v{module.version}  dof={module.dof}  "
                f"mass={module.total_mass:.3f}kg  ports: {ports}"
            )
    return EXIT_OK


def _cmd_show(args: argparse.Namespace, catalog: Catalog) -> int:
    try:
        module = catalog.get(args.module_id, args.version)
    except CatalogError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_FAILED

    print(f"{module.id} v{module.version} ({module.category})")
    if module.display_name:
        print(f"  {module.display_name}")
    if module.description:
        print(f"  {module.description.strip()}")
    print(f"  root_link: {module.root_link}")
    print(f"  links: {', '.join(link.name for link in module.links)}")
    print(f"  dof: {module.dof}   mass: {module.total_mass:.3f} kg")
    print("  ports:")
    for port in module.ports:
        note = f"  {port.description}" if port.description else ""
        print(f"    {port.name} [{port.role}/{port.interface}] on {port.link}{note}")
    for parameter in module.parameters:
        print(
            f"  parameter {parameter.name}: default={parameter.default} "
            f"range=[{parameter.min}, {parameter.max}] {parameter.unit}"
        )
    return EXIT_OK


def _cmd_index(args: argparse.Namespace, catalog: Catalog) -> int:
    del args
    path = write_catalog_index(catalog)
    print(f"wrote {path}")
    return EXIT_OK


def _cmd_digest(args: argparse.Namespace, catalog: Catalog) -> int:
    del args
    from urdfgen.catalog.digest import catalog_digest

    print(catalog_digest(catalog))
    return EXIT_OK


def _cmd_build_module(args: argparse.Namespace, catalog: Catalog) -> int:
    try:
        robot = expand_module(args.module_id, catalog, version=args.version)
    except (CatalogError, ExpansionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_FAILED
    return _emit(robot, catalog, args, source_text=f"build-module {args.module_id}")


def _cmd_build(args: argparse.Namespace, catalog: Catalog) -> int:
    spec = _read_spec(args.spec)
    if spec is None:
        return EXIT_USAGE

    findings = validate(spec, catalog)
    _print_findings(findings)
    if any(finding.is_error for finding in findings) and not args.skip_validation:
        print("error: validation failed; no URDF written", file=sys.stderr)
        return EXIT_FAILED

    try:
        robot = expand(spec, catalog)
    except ExpansionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_FAILED
    return _emit(robot, catalog, args, source_text=spec.name)


def _cmd_validate(args: argparse.Namespace, catalog: Catalog) -> int:
    spec = _read_spec(args.spec)
    if spec is None:
        return EXIT_USAGE

    findings = validate(spec, catalog)
    _print_findings(findings)
    errors = sum(1 for finding in findings if finding.is_error)
    warnings = len(findings) - errors
    print(f"{errors} error(s), {warnings} warning(s)")
    return EXIT_FAILED if errors else EXIT_OK


def _cmd_assemble(args: argparse.Namespace, catalog: Catalog) -> int:
    outcome = _run_assembly(args, catalog)
    if outcome is None:
        return EXIT_FAILED

    _report_outcome(outcome)
    if outcome.spec is None:
        return EXIT_FAILED

    document = outcome.spec.model_dump_json(indent=2, exclude_none=True) + "\n"
    if args.output is None:
        sys.stdout.write(document)
    else:
        args.output.write_text(document, encoding="utf-8")
        print(f"wrote {args.output}", file=sys.stderr)

    # A spec that still has errors is written out anyway -- it is the starting
    # point for a manual fix -- but the exit code says it is not usable yet.
    return EXIT_OK if outcome.succeeded else EXIT_FAILED


def _cmd_generate(args: argparse.Namespace, catalog: Catalog) -> int:
    outcome = _run_assembly(args, catalog)
    if outcome is None:
        return EXIT_FAILED

    _report_outcome(outcome)
    if outcome.spec is None or not outcome.succeeded:
        print(
            "error: no valid assembly was produced; run 'urdfgen assemble' to get "
            "the partial spec and edit it by hand",
            file=sys.stderr,
        )
        return EXIT_FAILED

    try:
        robot = expand(outcome.spec, catalog)
    except ExpansionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_FAILED
    return _emit(robot, catalog, args, source_text=args.text)


def _cmd_preview(args: argparse.Namespace, catalog: Catalog) -> int:
    del catalog  # the URDF is self-contained; the catalog is not consulted
    try:
        configuration = _parse_joints(args.joint)
        svg = render_file(
            str(args.urdf),
            configuration=configuration,
            view=args.view,
            width=args.width,
            title=args.title,
            show_grid=not args.no_grid,
            show_axes=not args.no_axes,
        )
    except PreviewError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_FAILED

    if args.output is None:
        sys.stdout.write(svg)
    else:
        args.output.write_text(svg, encoding="utf-8")
        print(f"wrote {args.output}", file=sys.stderr)
    return EXIT_OK


def _parse_joints(values: list[str]) -> dict[str, float]:
    """Read repeated ``--joint name=value`` flags."""
    configuration: dict[str, float] = {}
    for item in values:
        name, separator, raw = item.partition("=")
        if not separator or not name.strip():
            msg = f"expected NAME=VALUE, got {item!r}"
            raise PreviewError(msg)
        try:
            configuration[name.strip()] = float(raw)
        except ValueError as exc:
            msg = f"joint {name.strip()!r} needs a number, got {raw!r}"
            raise PreviewError(msg) from exc
    return configuration


def _run_assembly(args: argparse.Namespace, catalog: Catalog) -> AssemblyOutcome | None:
    """Set up the client and run the repair loop. ``None`` means it never started."""
    try:
        config = load_config(model=args.model, base_url=args.base_url)
    except LLMConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return None

    print(f"using {config.model} at {config.base_url}", file=sys.stderr)
    try:
        with LocalLLMClient(config) as client:
            return assemble_with_repair(args.text, catalog, client, max_attempts=args.attempts)
    except LLMError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return None


def _report_outcome(outcome: AssemblyOutcome) -> None:
    if outcome.spec is not None and outcome.spec.notes:
        print(f"notes: {outcome.spec.notes}", file=sys.stderr)
    if not outcome.schema_enforced:
        print(
            "warning: the server did not constrain generation to the JSON schema, "
            "so module ids were only suggested, not enforced",
            file=sys.stderr,
        )
    _print_findings(outcome.findings)
    print(f"{summarise(outcome.findings)} after {outcome.attempts} attempt(s)", file=sys.stderr)


# --- helpers --------------------------------------------------------------


def _read_spec(path: Path) -> AssemblySpec | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        print(f"error: cannot read {path}: {exc}", file=sys.stderr)
        return None
    except json.JSONDecodeError as exc:
        print(f"error: {path} is not valid JSON: {exc}", file=sys.stderr)
        return None
    try:
        return AssemblySpec.model_validate(raw)
    except PydanticValidationError as exc:
        print(f"error: {path} is not a valid AssemblySpec:\n{exc}", file=sys.stderr)
        return None


def _emit(
    robot: ExpandedRobot, catalog: Catalog, args: argparse.Namespace, source_text: str
) -> int:
    xml = build_urdf_string(robot, catalog, package_name=args.package, source_text=source_text)
    if args.output is None:
        sys.stdout.write(xml)
    else:
        args.output.write_text(xml, encoding="utf-8")
        print(
            f"wrote {args.output}: {len(robot.links)} links, {len(robot.joints)} joints "
            f"({len(robot.movable_joints)} movable)",
            file=sys.stderr,
        )
    return EXIT_OK


def _print_findings(findings: list[ValidationError]) -> None:
    for finding in findings:
        print(f"  [{finding.severity}] {finding.code} at {finding.path}: {finding.message}")


if __name__ == "__main__":
    raise SystemExit(main())
