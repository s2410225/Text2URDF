"""Static, headless previews of a generated URDF.

The web viewer in phase 4 will be interactive; this is the complement to it --
a picture that can be produced in CI, embedded in documentation, or looked at
over SSH, with no browser, no GPU and no display.

Input is a URDF *file*, not an internal data structure, so what gets drawn is
exactly what the builder emitted.
"""

from urdfgen.preview.svg import (
    VIEWS,
    PreviewError,
    render,
    render_file,
)

__all__ = ["VIEWS", "PreviewError", "render", "render_file"]
