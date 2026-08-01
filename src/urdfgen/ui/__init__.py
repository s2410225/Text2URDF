"""A standalone assembly screen, generated from the catalog.

The page is not hand-written: the catalog payload is injected into a template,
so changing a module changes the UI. Writing the data into the HTML by hand
would fork the single source of truth, which is the same mistake as letting the
model invent geometry.

It is self-contained -- no CDN, no build step, no server -- so it opens from a
file:// URL or as a published artifact. Phase 4's React and three.js viewer will
supersede it; this establishes the interaction and proves the payload is enough
to drive one.

The page never writes URDF. It produces an ``AssemblySpec``, exactly like the
LLM layer, and ``urdfgen build`` turns that into XML.
"""

from urdfgen.ui.page import DEFAULT_UNIT_NAME, render_page, write_page

__all__ = ["DEFAULT_UNIT_NAME", "render_page", "write_page"]
