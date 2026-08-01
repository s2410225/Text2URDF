"""Write a :class:`~urdfgen.preview.scene.Scene` out as a PNG.

Pillow's polygon fill has no antialiasing, so the scene is drawn at several
times the requested size and then resampled down. That is slower than a real
rasteriser and entirely adequate here: a robot is a few hundred flat polygons.

The output is deterministic -- the same scene at the same width produces the
same bytes -- so a rendering can be committed and diffed.
"""

from __future__ import annotations

import io

from PIL import Image, ImageDraw, ImageFont

from urdfgen.preview.scene import Line, Rgb, Scene

SUPERSAMPLE = 3
"""Draw at this multiple of the output size, then resample down.

Three is where the staircase on a cylinder's silhouette stops being visible;
higher costs memory for no gain a reader would notice.
"""

MAX_PIXELS = 40_000_000
"""Ceiling on the supersampled buffer, to refuse an accidental huge render."""

MIN_STROKE = 1


class RasterError(Exception):
    """Raised when a scene cannot be rasterised at the requested size."""


def to_png(scene: Scene, width: int, *, supersample: int = SUPERSAMPLE) -> bytes:
    """Render ``scene`` at ``width`` pixels and return PNG bytes."""
    height = scene.height_for(width)
    if width * height * supersample * supersample > MAX_PIXELS:
        msg = (
            f"rendering {width}x{height} at {supersample}x supersampling would need "
            f"more than {MAX_PIXELS:,} pixels; reduce --width"
        )
        raise RasterError(msg)

    canvas = Image.new(
        "RGB",
        (width * supersample, height * supersample),
        _rgb255(scene.background),
    )
    draw = ImageDraw.Draw(canvas)

    # View-box units to supersampled pixels.
    scale = scene.scale_for(width) * supersample
    origin = (scene.view_box[0], scene.view_box[1])

    def place(point: tuple[float, float]) -> tuple[float, float]:
        return ((point[0] - origin[0]) * scale, (point[1] - origin[1]) * scale)

    def draw_lines(lines: tuple[Line, ...]) -> None:
        for line in lines:
            draw.line(
                [place(line.start), place(line.end)],
                fill=_rgb255(line.color),
                width=_stroke(line.width, supersample),
            )

    draw_lines(scene.lines)

    edge_width = _stroke(scene.edge_width, supersample)
    for polygon in scene.polygons:
        points = [place(point) for point in polygon.points]
        draw.polygon(
            points,
            fill=_rgb255(polygon.fill),
            outline=_rgb255(polygon.edge),
            width=edge_width,
        )

    draw_lines(scene.overlays)

    for label in scene.labels:
        size = max(1, round(label.size * supersample))
        x, y = place(label.position)
        draw.text(
            (x, y - label.offset * supersample),
            label.text,
            fill=_rgb255(label.color),
            font=ImageFont.load_default(size=size),
            anchor="ms",
        )

    if supersample > 1:
        canvas = canvas.resize((width, height), Image.Resampling.LANCZOS)

    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def _rgb255(color: Rgb) -> tuple[int, int, int]:
    return tuple(round(min(1.0, max(0.0, channel)) * 255) for channel in color)  # type: ignore[return-value]


def _stroke(width_units: float, supersample: int) -> int:
    return max(MIN_STROKE, round(width_units * supersample))
