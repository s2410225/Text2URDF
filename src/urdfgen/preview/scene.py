"""Turn a parsed URDF into a flat 2D scene, independent of any output format.

Every shape in the catalog is a primitive, so the robot can be projected
analytically instead of rasterised: tessellate each box, cylinder and sphere
into flat faces, transform them into world space, project them orthographically
and sort them back to front. That needs no OpenGL, no display and no image
library, which is what makes it usable from CI.

Forward kinematics comes from ``yourdfpy`` rather than from
:mod:`urdfgen.builder.transform`. Reusing our own transform code would make the
picture agree with the builder by construction, which is precisely what a
preview must not do -- an independent path is what lets the drawing catch a
mistake.

What comes out is a :class:`Scene`: polygons already in paint order, lines and
labels, all in view-box units with colours as floats. Both the SVG and the PNG
backend consume exactly this, so the two can never drift apart.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import yourdfpy

Vec3 = tuple[float, float, float]
Point = tuple[float, float]
Rgb = tuple[float, float, float]
Matrix = npt.NDArray[np.float64]

VIEWS: dict[str, Vec3] = {
    "iso": (1.0, -1.0, 0.75),
    "front": (1.0, 0.0, 0.0),
    "side": (0.0, -1.0, 0.0),
    "top": (0.0, 0.0, 1.0),
}
"""Named camera directions, pointing from the robot toward the viewer.

``front``, ``side`` and ``top`` are exactly axis-aligned. Tilting them slightly
would look a little more three-dimensional, but it would also destroy the
property that makes them useful: a robot symmetric about a plane projects
symmetrically, so an asymmetry in the drawing is an asymmetry in the robot.
"""

CYLINDER_SEGMENTS = 20
SPHERE_SEGMENTS = 16
SPHERE_RINGS = 8

LIGHT_DIRECTION: Vec3 = (0.35, -0.6, 0.8)
AMBIENT = 0.42
"""Floor brightness, so faces turned away from the light stay readable."""

EDGE_DARKENING = 0.72
"""Edges are the face's own colour, darkened.

A single translucent black would need alpha compositing, which the raster
backend cannot do cheaply -- and tinting each edge with its own face reads
better anyway.
"""

DEFAULT_COLOR: Rgb = (0.62, 0.64, 0.67)
BACKGROUND: Rgb = (0.984, 0.984, 0.992)
GRID_COLOR: Rgb = (0.835, 0.847, 0.871)
LABEL_COLOR: Rgb = (0.290, 0.310, 0.341)
AXIS_COLORS: dict[str, Rgb] = {
    "x": (0.839, 0.271, 0.271),
    "y": (0.247, 0.569, 0.259),
    "z": (0.227, 0.431, 0.659),
}

MARGIN_FRACTION = 0.06
VERTICAL_TOLERANCE = 1e-6

GRID_WIDTH_UNITS = 1.0
AXIS_WIDTH_UNITS = 2.0
EDGE_WIDTH_UNITS = 0.5
LABEL_SIZE_UNITS = 16.0
LABEL_BASELINE_UNITS = 6.0
"""Sizes in units of one pixel at the requested output width.

Keeping them relative means an image asked for at 400 px and one at 2000 px
have the same visual weight instead of hairlines at one end and slabs at the
other.
"""


class PreviewError(Exception):
    """Raised when a robot cannot be drawn as asked."""


@dataclass(frozen=True)
class Face:
    """One flat polygon in world space, before projection."""

    points: tuple[Vec3, ...]
    color: Rgb
    link: str


@dataclass(frozen=True)
class Polygon:
    """A projected face. ``link`` survives so backends can label their output."""

    points: tuple[Point, ...]
    fill: Rgb
    edge: Rgb
    link: str


@dataclass(frozen=True)
class Line:
    start: Point
    end: Point
    color: Rgb
    width: float
    """In units of one output pixel."""


@dataclass(frozen=True)
class Label:
    text: str
    position: Point
    """Horizontally centred on this point, and sitting ``offset`` above it."""

    size: float
    """Cap height in units of one output pixel."""

    offset: float
    """How far above ``position`` the baseline goes, in output pixels."""

    color: Rgb


@dataclass(frozen=True)
class Scene:
    """A drawing, ready to be written out in any format."""

    view_box: tuple[float, float, float, float]
    background: Rgb

    lines: tuple[Line, ...]
    """Drawn under the robot: the ground grid."""

    polygons: tuple[Polygon, ...]
    """Already sorted far to near, so painting them in order is correct."""

    overlays: tuple[Line, ...]
    """Drawn over the robot: the origin axes, which are an annotation rather
    than part of the scene and would otherwise be buried inside the body."""

    labels: tuple[Label, ...]
    edge_width: float

    @property
    def aspect(self) -> float:
        return self.view_box[3] / self.view_box[2]

    def height_for(self, width: int) -> int:
        return max(1, round(width * self.aspect))

    def scale_for(self, width: int) -> float:
        """View-box units to pixels."""
        return width / self.view_box[2]


def build_scene(
    robot: yourdfpy.URDF,
    *,
    configuration: Mapping[str, float] | None = None,
    view: str = "iso",
    title: str | None = None,
    show_grid: bool = True,
    show_axes: bool = True,
) -> Scene:
    """Project ``robot`` in the given joint configuration into a 2D scene."""
    if view not in VIEWS:
        known = ", ".join(sorted(VIEWS))
        msg = f"unknown view {view!r}; choose one of {known}"
        raise PreviewError(msg)

    apply_configuration(robot, configuration)
    faces = collect_faces(robot)
    if not faces:
        msg = "the robot has no visual geometry to draw"
        raise PreviewError(msg)

    camera = _camera(VIEWS[view])
    grid = _grid_lines(faces) if show_grid else []
    axes = _axis_lines(faces) if show_axes else []

    projected = [camera.project(point) for face in faces for point in face.points]
    projected += [camera.project(point) for line in grid for point in line]
    projected += [camera.project(point) for _, line in axes for point in line]
    view_box = _fit(projected)

    polygons = tuple(
        Polygon(
            points=tuple(camera.project(point) for point in face.points),
            fill=_shade(face, camera),
            edge=_darken(_shade(face, camera)),
            link=face.link,
        )
        for face in sorted(faces, key=lambda item: _centroid_depth(camera, item))
    )

    lines = tuple(
        Line(camera.project(a), camera.project(b), GRID_COLOR, GRID_WIDTH_UNITS) for a, b in grid
    )
    overlays = tuple(
        Line(camera.project(a), camera.project(b), AXIS_COLORS[axis], AXIS_WIDTH_UNITS)
        for axis, (a, b) in axes
    )

    labels: tuple[Label, ...] = ()
    if title:
        labels = (
            Label(
                text=title,
                position=(view_box[0] + view_box[2] / 2, view_box[1] + view_box[3]),
                size=LABEL_SIZE_UNITS,
                offset=LABEL_BASELINE_UNITS,
                color=LABEL_COLOR,
            ),
        )

    return Scene(
        view_box=view_box,
        background=BACKGROUND,
        lines=lines,
        polygons=polygons,
        overlays=overlays,
        labels=labels,
        edge_width=EDGE_WIDTH_UNITS,
    )


# --- kinematics -----------------------------------------------------------


def apply_configuration(robot: yourdfpy.URDF, configuration: Mapping[str, float] | None) -> None:
    """Move the robot into the requested pose, rejecting unknown joints.

    A typo in a joint name would otherwise be silently ignored and the drawing
    would show the default pose while claiming to show something else.
    """
    actuated = list(robot.actuated_joint_names)
    if not configuration:
        robot.update_cfg(np.zeros(len(actuated)))
        return

    unknown = sorted(set(configuration) - set(actuated))
    if unknown:
        msg = (
            f"unknown or non-actuated joint(s) {unknown}; "
            f"this robot moves: {', '.join(actuated) or '(none)'}"
        )
        raise PreviewError(msg)

    robot.update_cfg(np.array([float(configuration.get(name, 0.0)) for name in actuated]))


def collect_faces(robot: yourdfpy.URDF) -> list[Face]:
    """Tessellate every visual in the robot into world-space polygons."""
    colors = _material_colors(robot)
    faces: list[Face] = []
    for name, link in robot.link_map.items():
        world = _link_transform(robot, name)
        for visual in link.visuals:
            placement = world if visual.origin is None else world @ np.asarray(visual.origin)
            color = _visual_color(visual, colors)
            faces.extend(_tessellate(visual.geometry, placement, color, name))
    return faces


def _link_transform(robot: yourdfpy.URDF, link: str) -> Matrix:
    try:
        return np.asarray(robot.get_transform(frame_to=link, frame_from=robot.base_link))
    except Exception as exc:
        msg = f"cannot place link {link!r}: {exc}"
        raise PreviewError(msg) from exc


# --- colours --------------------------------------------------------------


def _material_colors(robot: yourdfpy.URDF) -> dict[str, Rgb]:
    colors: dict[str, Rgb] = {}
    for material in robot.robot.materials:
        if material.color is not None:
            colors[material.name] = _rgb(material.color.rgba)
    return colors


def _visual_color(visual: object, colors: Mapping[str, Rgb]) -> Rgb:
    material = getattr(visual, "material", None)
    if material is None:
        return DEFAULT_COLOR
    if material.color is not None:
        return _rgb(material.color.rgba)
    return colors.get(material.name, DEFAULT_COLOR)


def _rgb(rgba: object) -> Rgb:
    values = np.asarray(rgba, dtype=float)
    return (float(values[0]), float(values[1]), float(values[2]))


def _darken(color: Rgb) -> Rgb:
    return (
        color[0] * EDGE_DARKENING,
        color[1] * EDGE_DARKENING,
        color[2] * EDGE_DARKENING,
    )


# --- tessellation ---------------------------------------------------------


def _tessellate(geometry: object, transform: Matrix, color: Rgb, link: str) -> list[Face]:
    box = getattr(geometry, "box", None)
    cylinder = getattr(geometry, "cylinder", None)
    sphere = getattr(geometry, "sphere", None)

    if box is not None:
        size = np.asarray(box.size, dtype=float)
        polygons = _box_polygons((float(size[0]), float(size[1]), float(size[2])))
    elif cylinder is not None:
        polygons = _cylinder_polygons(float(cylinder.radius), float(cylinder.length))
    elif sphere is not None:
        polygons = _sphere_polygons(float(sphere.radius))
    else:
        # Mesh geometry: the catalog has none yet, and drawing a placeholder box
        # would misrepresent the robot's shape.
        return []

    return [
        Face(points=tuple(_apply(transform, point) for point in polygon), color=color, link=link)
        for polygon in polygons
    ]


def _box_polygons(size: Vec3) -> list[tuple[Vec3, ...]]:
    x, y, z = (value / 2.0 for value in size)
    corners = [
        (-x, -y, -z),
        (x, -y, -z),
        (x, y, -z),
        (-x, y, -z),
        (-x, -y, z),
        (x, -y, z),
        (x, y, z),
        (-x, y, z),
    ]
    quads = [
        (0, 1, 2, 3),  # bottom
        (4, 5, 6, 7),  # top
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    ]
    return [tuple(corners[index] for index in quad) for quad in quads]


def _cylinder_polygons(radius: float, length: float) -> list[tuple[Vec3, ...]]:
    """A cylinder about Z, approximated by a prism plus two caps."""
    half = length / 2.0
    ring = [
        (
            radius * math.cos(2.0 * math.pi * i / CYLINDER_SEGMENTS),
            radius * math.sin(2.0 * math.pi * i / CYLINDER_SEGMENTS),
        )
        for i in range(CYLINDER_SEGMENTS)
    ]

    polygons: list[tuple[Vec3, ...]] = []
    for index, (cx, cy) in enumerate(ring):
        nx, ny = ring[(index + 1) % CYLINDER_SEGMENTS]
        polygons.append(((cx, cy, -half), (nx, ny, -half), (nx, ny, half), (cx, cy, half)))

    polygons.append(tuple((x, y, half) for x, y in ring))
    polygons.append(tuple((x, y, -half) for x, y in reversed(ring)))
    return polygons


def _sphere_polygons(radius: float) -> list[tuple[Vec3, ...]]:
    """A UV sphere. Coarse on purpose -- it only ever renders a foot pad."""

    def point(ring: int, segment: int) -> Vec3:
        phi = math.pi * ring / SPHERE_RINGS
        theta = 2.0 * math.pi * segment / SPHERE_SEGMENTS
        return (
            radius * math.sin(phi) * math.cos(theta),
            radius * math.sin(phi) * math.sin(theta),
            radius * math.cos(phi),
        )

    polygons: list[tuple[Vec3, ...]] = []
    for ring in range(SPHERE_RINGS):
        for segment in range(SPHERE_SEGMENTS):
            corners = (
                point(ring, segment),
                point(ring, segment + 1),
                point(ring + 1, segment + 1),
                point(ring + 1, segment),
            )
            # Collapse the degenerate quads at the poles into triangles.
            unique = tuple(dict.fromkeys(corners))
            if len(unique) >= 3:
                polygons.append(unique)
    return polygons


def _apply(transform: Matrix, point: Vec3) -> Vec3:
    result = transform @ np.array([point[0], point[1], point[2], 1.0])
    return (float(result[0]), float(result[1]), float(result[2]))


# --- projection -----------------------------------------------------------


@dataclass(frozen=True)
class _Camera:
    right: Matrix
    up: Matrix
    forward: Matrix
    """Points from the robot toward the viewer, so larger depth is nearer."""

    def project(self, point: Vec3) -> Point:
        vector = np.asarray(point, dtype=float)
        return (float(vector @ self.right), -float(vector @ self.up))

    def depth(self, point: Vec3) -> float:
        return float(np.asarray(point, dtype=float) @ self.forward)


def _camera(direction: Vec3) -> _Camera:
    forward = np.asarray(direction, dtype=float)
    forward = forward / np.linalg.norm(forward)

    # +Z is up on screen, except when looking straight down it -- there the
    # reference has to be something else, and +Y up matches the convention for
    # a plan view (X to the right, Y away from the viewer).
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(forward @ reference)) > 1.0 - VERTICAL_TOLERANCE:
        reference = np.array([0.0, 1.0, 0.0])

    right = np.cross(reference, forward)
    right = right / np.linalg.norm(right)
    return _Camera(right=right, up=np.cross(forward, right), forward=forward)


def _centroid_depth(camera: _Camera, face: Face) -> float:
    return sum(camera.depth(point) for point in face.points) / len(face.points)


def _shade(face: Face, camera: _Camera) -> Rgb:
    normal = _normal(face.points)
    if normal is None:
        intensity = 1.0
    else:
        if normal @ camera.forward < 0.0:
            normal = -normal
        light = np.asarray(LIGHT_DIRECTION, dtype=float)
        light = light / np.linalg.norm(light)
        intensity = AMBIENT + (1.0 - AMBIENT) * max(0.0, float(normal @ light))

    return (
        min(1.0, face.color[0] * intensity),
        min(1.0, face.color[1] * intensity),
        min(1.0, face.color[2] * intensity),
    )


def _normal(points: Sequence[Vec3]) -> Matrix | None:
    """Newell's method, which is stable for the near-degenerate cap polygons."""
    normal = np.zeros(3)
    for index, current in enumerate(points):
        following = points[(index + 1) % len(points)]
        normal[0] += (current[1] - following[1]) * (current[2] + following[2])
        normal[1] += (current[2] - following[2]) * (current[0] + following[0])
        normal[2] += (current[0] - following[0]) * (current[1] + following[1])
    length = float(np.linalg.norm(normal))
    return None if length < 1e-12 else normal / length


# --- scene furniture ------------------------------------------------------


def _bounds(faces: Iterable[Face]) -> tuple[Matrix, Matrix]:
    points = np.array([point for face in faces for point in face.points], dtype=float)
    return points.min(axis=0), points.max(axis=0)


def _grid_lines(faces: Sequence[Face]) -> list[tuple[Vec3, Vec3]]:
    """A floor grid at the robot's lowest point, to give the drawing a ground."""
    low, high = _bounds(faces)
    span = float(max(high[0] - low[0], high[1] - low[1])) * 0.75 + 0.05
    step = _grid_step(span * 2.0)
    floor = float(low[2])

    centre = ((low[0] + high[0]) / 2.0, (low[1] + high[1]) / 2.0)
    count = int(span / step) + 1

    lines: list[tuple[Vec3, Vec3]] = []
    for index in range(-count, count + 1):
        offset = index * step
        lines.append(
            (
                (centre[0] + offset, centre[1] - count * step, floor),
                (centre[0] + offset, centre[1] + count * step, floor),
            )
        )
        lines.append(
            (
                (centre[0] - count * step, centre[1] + offset, floor),
                (centre[0] + count * step, centre[1] + offset, floor),
            )
        )
    return lines


def _grid_step(span: float) -> float:
    """A round grid spacing that keeps the grid readable at any robot size."""
    for step in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0):
        if span / step <= 12.0:
            return step
    return 10.0


def _axis_lines(faces: Sequence[Face]) -> list[tuple[str, tuple[Vec3, Vec3]]]:
    low, high = _bounds(faces)
    length = float(max(high - low)) * 0.2
    return [
        ("x", ((0.0, 0.0, 0.0), (length, 0.0, 0.0))),
        ("y", ((0.0, 0.0, 0.0), (0.0, length, 0.0))),
        ("z", ((0.0, 0.0, 0.0), (0.0, 0.0, length))),
    ]


def _fit(projected: Sequence[Point]) -> tuple[float, float, float, float]:
    xs = [point[0] for point in projected]
    ys = [point[1] for point in projected]
    span_x = max(max(xs) - min(xs), 1e-6)
    span_y = max(max(ys) - min(ys), 1e-6)
    margin = max(span_x, span_y) * MARGIN_FRACTION

    return (
        min(xs) - margin,
        min(ys) - margin,
        span_x + 2 * margin,
        span_y + 2 * margin,
    )
