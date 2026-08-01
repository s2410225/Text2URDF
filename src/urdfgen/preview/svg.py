"""Write a :class:`~urdfgen.preview.scene.Scene` out as SVG.

Vector output, so it stays sharp at any zoom and diffs as text. All the
geometry was decided in :mod:`urdfgen.preview.scene`; this only chooses how to
spell it.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

from urdfgen.preview.scene import Line, Rgb, Scene

FONT_STACK = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"


def to_svg(scene: Scene, width: int) -> str:
    """Render ``scene`` at ``width`` pixels. Returns a complete SVG document."""
    x, y, box_width, box_height = scene.view_box
    unit = 1.0 / scene.scale_for(width)
    """One output pixel, expressed in view-box units."""

    svg = ET.Element(
        "svg",
        {
            "xmlns": "http://www.w3.org/2000/svg",
            "viewBox": " ".join(f"{value:.4f}" for value in scene.view_box),
            "width": str(width),
            "height": str(scene.height_for(width)),
        },
    )
    ET.SubElement(
        svg,
        "rect",
        {
            "x": f"{x:.4f}",
            "y": f"{y:.4f}",
            "width": f"{box_width:.4f}",
            "height": f"{box_height:.4f}",
            "fill": _hex(scene.background),
        },
    )

    def draw_lines(lines: tuple[Line, ...]) -> None:
        for line in lines:
            ET.SubElement(
                svg,
                "line",
                {
                    "x1": f"{line.start[0]:.5f}",
                    "y1": f"{line.start[1]:.5f}",
                    "x2": f"{line.end[0]:.5f}",
                    "y2": f"{line.end[1]:.5f}",
                    "stroke": _hex(line.color),
                    "stroke-width": f"{line.width * unit:.5f}",
                },
            )

    draw_lines(scene.lines)

    for polygon in scene.polygons:
        ET.SubElement(
            svg,
            "polygon",
            {
                "points": " ".join(f"{px:.5f},{py:.5f}" for px, py in polygon.points),
                "fill": _hex(polygon.fill),
                "stroke": _hex(polygon.edge),
                "stroke-width": f"{scene.edge_width * unit:.5f}",
                "stroke-linejoin": "round",
                "data-link": polygon.link,
            },
        )

    draw_lines(scene.overlays)

    for label in scene.labels:
        text = ET.SubElement(
            svg,
            "text",
            {
                "x": f"{label.position[0]:.4f}",
                "y": f"{label.position[1] - label.offset * unit:.4f}",
                "text-anchor": "middle",
                "font-family": FONT_STACK,
                "font-size": f"{label.size * unit:.4f}",
                "fill": _hex(label.color),
            },
        )
        text.text = label.text

    ET.indent(svg, space="  ")
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(svg, encoding="unicode") + "\n"


def _hex(color: Rgb) -> str:
    return "#" + "".join(f"{round(min(1.0, max(0.0, channel)) * 255):02x}" for channel in color)
