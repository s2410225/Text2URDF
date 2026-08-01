"""Static, headless previews of a generated URDF.

The web viewer in phase 4 will be interactive; this is the complement to it --
a picture that can be produced in CI, embedded in documentation, or looked at
over SSH, with no browser, no GPU and no display.

Input is a URDF *file*, not an internal data structure, so what gets drawn is
exactly what the builder emitted.

Two output formats share one projection: SVG stays sharp at any zoom and diffs
as text, PNG previews inline anywhere an image does. :func:`save` picks between
them from the output file's extension.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import yourdfpy

from urdfgen.preview.raster import RasterError, to_png
from urdfgen.preview.scene import (
    VIEWS,
    PreviewError,
    Scene,
    build_scene,
)
from urdfgen.preview.svg import to_svg

__all__ = [
    "FORMATS",
    "VIEWS",
    "PreviewError",
    "Scene",
    "build_scene",
    "format_for",
    "load",
    "render",
    "render_file",
    "render_png",
    "render_png_file",
    "save",
]

DEFAULT_WIDTH = 900

FORMATS: tuple[str, ...] = ("svg", "png")

_EXTENSIONS = {".svg": "svg", ".png": "png"}


def load(path: str | Path) -> yourdfpy.URDF:
    """Parse a URDF file. The only file read in this package."""
    try:
        return yourdfpy.URDF.load(str(path), load_meshes=False, load_collision_meshes=False)
    except Exception as exc:
        msg = f"cannot read {path} as a URDF: {exc}"
        raise PreviewError(msg) from exc


def format_for(path: str | Path) -> str:
    """The output format implied by ``path``'s extension."""
    suffix = Path(path).suffix.lower()
    try:
        return _EXTENSIONS[suffix]
    except KeyError:
        known = ", ".join(sorted(_EXTENSIONS))
        msg = f"cannot tell the format from {Path(path).name!r}; use one of {known}"
        raise PreviewError(msg) from None


def render(
    robot: yourdfpy.URDF,
    *,
    configuration: Mapping[str, float] | None = None,
    view: str = "iso",
    width: int = DEFAULT_WIDTH,
    title: str | None = None,
    show_grid: bool = True,
    show_axes: bool = True,
) -> str:
    """Draw ``robot`` as SVG."""
    _check_width(width)
    return to_svg(_scene(robot, configuration, view, title, show_grid, show_axes), width)


def render_png(
    robot: yourdfpy.URDF,
    *,
    configuration: Mapping[str, float] | None = None,
    view: str = "iso",
    width: int = DEFAULT_WIDTH,
    title: str | None = None,
    show_grid: bool = True,
    show_axes: bool = True,
) -> bytes:
    """Draw ``robot`` as a PNG."""
    _check_width(width)
    scene = _scene(robot, configuration, view, title, show_grid, show_axes)
    try:
        return to_png(scene, width)
    except RasterError as exc:
        raise PreviewError(str(exc)) from exc


def render_file(
    path: str | Path,
    *,
    configuration: Mapping[str, float] | None = None,
    view: str = "iso",
    width: int = DEFAULT_WIDTH,
    title: str | None = None,
    show_grid: bool = True,
    show_axes: bool = True,
) -> str:
    """Load a URDF and draw it as SVG."""
    robot = load(path)
    return render(
        robot,
        configuration=configuration,
        view=view,
        width=width,
        title=_title(title, robot),
        show_grid=show_grid,
        show_axes=show_axes,
    )


def render_png_file(
    path: str | Path,
    *,
    configuration: Mapping[str, float] | None = None,
    view: str = "iso",
    width: int = DEFAULT_WIDTH,
    title: str | None = None,
    show_grid: bool = True,
    show_axes: bool = True,
) -> bytes:
    """Load a URDF and draw it as a PNG."""
    robot = load(path)
    return render_png(
        robot,
        configuration=configuration,
        view=view,
        width=width,
        title=_title(title, robot),
        show_grid=show_grid,
        show_axes=show_axes,
    )


def save(
    urdf: str | Path,
    output: str | Path,
    *,
    fmt: str | None = None,
    configuration: Mapping[str, float] | None = None,
    view: str = "iso",
    width: int = DEFAULT_WIDTH,
    title: str | None = None,
    show_grid: bool = True,
    show_axes: bool = True,
) -> str:
    """Draw ``urdf`` into ``output``, choosing the format from its extension.

    ``fmt`` overrides that. Returns the format actually written.
    """
    chosen = fmt or format_for(output)
    if chosen not in FORMATS:
        msg = f"unknown format {chosen!r}; use one of {', '.join(FORMATS)}"
        raise PreviewError(msg)

    options = {
        "configuration": configuration,
        "view": view,
        "width": width,
        "title": title,
        "show_grid": show_grid,
        "show_axes": show_axes,
    }
    if chosen == "png":
        Path(output).write_bytes(render_png_file(urdf, **options))  # type: ignore[arg-type]
    else:
        Path(output).write_text(render_file(urdf, **options), encoding="utf-8")  # type: ignore[arg-type]
    return chosen


def _check_width(width: int) -> None:
    if width <= 0:
        msg = f"width must be positive, got {width}"
        raise PreviewError(msg)


def _scene(
    robot: yourdfpy.URDF,
    configuration: Mapping[str, float] | None,
    view: str,
    title: str | None,
    show_grid: bool,
    show_axes: bool,
) -> Scene:
    return build_scene(
        robot,
        configuration=configuration,
        view=view,
        title=title,
        show_grid=show_grid,
        show_axes=show_axes,
    )


def _title(title: str | None, robot: yourdfpy.URDF) -> str | None:
    """An explicit title wins; otherwise fall back to the robot's own name."""
    return robot.robot.name if title is None else title
