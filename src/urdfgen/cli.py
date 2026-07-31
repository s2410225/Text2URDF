"""Command line entry point for the deterministic half of the pipeline.

Text-to-assembly (the LLM layer) is not wired up here yet; these commands work
on catalog modules and hand-written ``AssemblySpec`` documents.
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
from urdfgen.validation import validate
from urdfgen.validation.result import ValidationError

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

    return parser


def _add_build_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-o", "--output", type=Path, default=None, help="output file (default: stdout)"
    )
    parser.add_argument(
        "--package", default=DEFAULT_PACKAGE_NAME, help="ROS package name used in mesh URIs"
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
