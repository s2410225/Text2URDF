"""Preview renderer tests.

How a drawing *looks* is not something a test can judge, so these check the
geometry instead: that every link is drawn, that the projection preserves the
robot's symmetry, and that moving a joint moves exactly the links downstream of
it and nothing else.
"""

from __future__ import annotations

import io
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
import yourdfpy
from PIL import Image

from conftest import EXPECTED
from urdfgen.preview import (
    VIEWS,
    PreviewError,
    format_for,
    render,
    render_file,
    render_png_file,
    save,
)

QUADRUPED = str(EXPECTED / "quadruped_a.urdf")
LEG = str(EXPECTED / "leg_3dof_left_v1.urdf")

SVG_NS = "{http://www.w3.org/2000/svg}"


def polygons(svg: str) -> list[ET.Element]:
    return list(ET.fromstring(svg).iter(f"{SVG_NS}polygon"))


def links_drawn(svg: str) -> set[str]:
    return {polygon.attrib["data-link"] for polygon in polygons(svg)}


def points_of(svg: str, link: str) -> list[tuple[float, float]]:
    """Every projected vertex belonging to ``link``, in drawing order."""
    result: list[tuple[float, float]] = []
    for polygon in polygons(svg):
        if polygon.attrib["data-link"] != link:
            continue
        for pair in polygon.attrib["points"].split():
            x, _, y = pair.partition(",")
            result.append((float(x), float(y)))
    return result


def centroid(svg: str, link: str) -> tuple[float, float]:
    points = points_of(svg, link)
    assert points, f"link {link!r} was not drawn"
    return (
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
    )


# --- output shape ---------------------------------------------------------


def test_output_is_well_formed_svg() -> None:
    root = ET.fromstring(render_file(QUADRUPED))
    assert root.tag == f"{SVG_NS}svg"
    assert root.attrib["viewBox"]


def test_every_link_is_drawn() -> None:
    """All 17 links of the quadruped must appear, not just the ones in front."""
    robot = yourdfpy.URDF.load(QUADRUPED, load_meshes=False, load_collision_meshes=False)
    assert links_drawn(render_file(QUADRUPED)) == set(robot.link_map)


def test_single_module_renders() -> None:
    assert links_drawn(render_file(LEG)) == {
        "leg_3dof_left_v1_hip_bracket",
        "leg_3dof_left_v1_hip_yoke",
        "leg_3dof_left_v1_thigh",
        "leg_3dof_left_v1_shank",
    }


def test_colours_come_from_the_urdf_materials() -> None:
    """The thigh is accent_orange (0.88, 0.42, 0.12), so it must render warm."""
    fills = {
        polygon.attrib["fill"]
        for polygon in polygons(render_file(QUADRUPED))
        if polygon.attrib["data-link"] == "leg_fl_thigh"
    }
    for fill in fills:
        red, green, blue = (int(fill[i : i + 2], 16) for i in (1, 3, 5))
        assert red > green > blue


def test_title_defaults_to_the_robot_name() -> None:
    assert "quadruped_a" in render_file(QUADRUPED)
    assert "custom caption" in render_file(QUADRUPED, title="custom caption")


def test_grid_and_axes_can_be_turned_off() -> None:
    bare = render_file(QUADRUPED, show_grid=False, show_axes=False)
    assert len(list(ET.fromstring(bare).iter(f"{SVG_NS}line"))) == 0
    assert len(list(ET.fromstring(render_file(QUADRUPED)).iter(f"{SVG_NS}line"))) > 0


# --- projection -----------------------------------------------------------


def test_projection_preserves_left_right_symmetry() -> None:
    """The quadruped is symmetric about y=0, so the side view must be too.

    Viewed along -Y, mirrored links land on top of each other. A sign error
    anywhere in the build chain would show up as a broken pair here.
    """
    svg = render_file(QUADRUPED, view="side")
    for left, right in (("leg_fl_shank", "leg_fr_shank"), ("leg_rl_shank", "leg_rr_shank")):
        assert centroid(svg, left) == pytest.approx(centroid(svg, right), abs=1e-6)


def test_the_four_legs_form_a_rectangle_from_above() -> None:
    """Seen from above the feet must sit on the corners of a rectangle.

    In the plan view screen X is world X and screen Y is -world Y, so the
    front pair shares a screen X and the left pair shares a screen Y.
    """
    svg = render_file(QUADRUPED, view="top")
    feet = {leg: centroid(svg, f"{leg}_shank") for leg in ("leg_fl", "leg_fr", "leg_rl", "leg_rr")}

    assert feet["leg_fl"][0] == pytest.approx(feet["leg_fr"][0], abs=1e-6)
    assert feet["leg_rl"][0] == pytest.approx(feet["leg_rr"][0], abs=1e-6)
    assert feet["leg_fl"][1] == pytest.approx(feet["leg_rl"][1], abs=1e-6)
    assert feet["leg_fr"][1] == pytest.approx(feet["leg_rr"][1], abs=1e-6)

    # And it is a rectangle with area, not four points on top of each other.
    assert feet["leg_fl"][0] != pytest.approx(feet["leg_rl"][0], abs=1e-3)
    assert feet["leg_fl"][1] != pytest.approx(feet["leg_fr"][1], abs=1e-3)


@pytest.mark.parametrize("view", sorted(VIEWS))
def test_every_view_draws_the_whole_robot(view: str) -> None:
    assert len(links_drawn(render_file(QUADRUPED, view=view))) == 17


def test_views_differ() -> None:
    assert render_file(QUADRUPED, view="top") != render_file(QUADRUPED, view="front")


def test_unknown_view_is_refused() -> None:
    with pytest.raises(PreviewError, match="unknown view"):
        render_file(QUADRUPED, view="behind")


# --- articulation ---------------------------------------------------------


def test_bending_a_knee_moves_only_that_shank() -> None:
    """Everything upstream of the joint must stay exactly where it was."""
    rest = render_file(QUADRUPED)
    bent = render_file(QUADRUPED, configuration={"leg_fl_knee_pitch": -1.2})

    assert centroid(bent, "leg_fl_shank") != pytest.approx(centroid(rest, "leg_fl_shank"), abs=1e-6)
    for untouched in ("body_chassis", "leg_fl_thigh", "leg_fr_shank", "leg_rr_shank"):
        assert centroid(bent, untouched) == pytest.approx(centroid(rest, untouched), abs=1e-9)


def test_the_hip_carries_the_whole_leg_with_it() -> None:
    rest = render_file(QUADRUPED)
    swung = render_file(QUADRUPED, configuration={"leg_fl_hip_pitch": 0.8})

    for moved in ("leg_fl_thigh", "leg_fl_shank"):
        assert centroid(swung, moved) != pytest.approx(centroid(rest, moved), abs=1e-6)
    assert centroid(swung, "leg_fl_hip_bracket") == pytest.approx(
        centroid(rest, "leg_fl_hip_bracket"), abs=1e-9
    )


def test_mirrored_legs_need_opposite_signs_for_the_same_pose() -> None:
    """The left and right leg modules are mirror images (spec 5.4).

    So the same physical stance is commanded with opposite joint values, and
    the result stays symmetric in the side view. This is the property the whole
    mirroring convention rests on, seen end to end through the built URDF.
    """
    svg = render_file(
        QUADRUPED,
        view="side",
        configuration={"leg_fl_knee_pitch": -1.2, "leg_fr_knee_pitch": 1.2},
    )
    assert centroid(svg, "leg_fl_shank") == pytest.approx(centroid(svg, "leg_fr_shank"), abs=1e-6)


def test_unknown_joint_is_refused() -> None:
    """A typo must fail rather than quietly render the default pose."""
    with pytest.raises(PreviewError, match="unknown or non-actuated"):
        render_file(QUADRUPED, configuration={"leg_fl_knee": 1.0})


def test_fixed_joints_cannot_be_driven() -> None:
    with pytest.raises(PreviewError, match="unknown or non-actuated"):
        render_file(QUADRUPED, configuration={"body_leg_mount_fl_to_leg_fl": 1.0})


# --- determinism ----------------------------------------------------------


def test_the_same_request_renders_identically() -> None:
    first = render_file(QUADRUPED, configuration={"leg_fl_knee_pitch": -0.5})
    second = render_file(QUADRUPED, configuration={"leg_fl_knee_pitch": -0.5})
    assert first == second


def test_an_omitted_joint_is_the_same_as_zero() -> None:
    explicit = render_file(QUADRUPED, configuration={"leg_fl_knee_pitch": 0.0})
    assert render_file(QUADRUPED) == explicit


# --- failure modes --------------------------------------------------------


def test_a_missing_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(PreviewError, match="cannot read"):
        render_file(str(tmp_path / "absent.urdf"))


def test_a_non_urdf_file_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("this is not a robot", encoding="utf-8")
    with pytest.raises(PreviewError, match="cannot read"):
        render_file(str(path))


def test_zero_width_is_refused() -> None:
    robot = yourdfpy.URDF.load(QUADRUPED, load_meshes=False, load_collision_meshes=False)
    with pytest.raises(PreviewError, match="width must be positive"):
        render(robot, width=0)


def test_a_robot_without_visuals_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "bare.urdf"
    path.write_text('<robot name="bare"><link name="only"/></robot>', encoding="utf-8")
    with pytest.raises(PreviewError, match="no visual geometry"):
        render_file(str(path))


# --- PNG output -----------------------------------------------------------

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def test_png_is_a_real_png() -> None:
    data = render_png_file(QUADRUPED)
    assert data.startswith(PNG_SIGNATURE)
    with Image.open(io.BytesIO(data)) as image:
        assert image.format == "PNG"


def test_png_honours_the_requested_width() -> None:
    with Image.open(io.BytesIO(render_png_file(QUADRUPED, width=640))) as image:
        assert image.width == 640


def test_png_matches_the_svg_aspect_ratio() -> None:
    """Both backends read the same scene, so they must frame it identically."""
    root = ET.fromstring(render_file(QUADRUPED, width=640))
    _, _, box_width, box_height = (float(v) for v in root.attrib["viewBox"].split())

    with Image.open(io.BytesIO(render_png_file(QUADRUPED, width=640))) as image:
        assert image.height == pytest.approx(640 * box_height / box_width, abs=1)
        assert (image.width, image.height) == (
            int(root.attrib["width"]),
            int(root.attrib["height"]),
        )


def test_png_carries_the_catalog_colours() -> None:
    """The thighs are accent_orange; that colour has to survive to the pixels."""
    with Image.open(io.BytesIO(render_png_file(QUADRUPED, width=400))) as image:
        pixels = np.asarray(image.convert("RGB"), dtype=int)

    red, green, blue = pixels[..., 0], pixels[..., 1], pixels[..., 2]
    warm = ((red > 120) & (red > green + 40) & (green > blue + 10) & (blue < 110)).sum()
    assert warm > 200, "no accent-orange region found in the render"


def test_png_is_deterministic() -> None:
    """Same robot, same pose, same width: byte-identical, so it can be diffed."""
    first = render_png_file(QUADRUPED, configuration={"leg_fl_knee_pitch": -0.5})
    second = render_png_file(QUADRUPED, configuration={"leg_fl_knee_pitch": -0.5})
    assert first == second


def test_png_changes_with_the_pose() -> None:
    assert render_png_file(QUADRUPED) != render_png_file(
        QUADRUPED, configuration={"leg_fl_knee_pitch": -1.2}
    )


def test_png_rejects_a_bad_joint_the_same_way_svg_does() -> None:
    with pytest.raises(PreviewError, match="unknown or non-actuated"):
        render_png_file(QUADRUPED, configuration={"nope": 1.0})


def test_an_absurd_width_is_refused() -> None:
    """A typo in --width should not try to allocate tens of gigabytes."""
    with pytest.raises(PreviewError, match="reduce --width"):
        render_png_file(QUADRUPED, width=100_000)


# --- format selection -----------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [("robot.png", "png"), ("robot.svg", "svg"), ("PATH/Robot.PNG", "png")],
)
def test_format_comes_from_the_extension(name: str, expected: str) -> None:
    assert format_for(name) == expected


def test_an_unknown_extension_is_refused() -> None:
    with pytest.raises(PreviewError, match="cannot tell the format"):
        format_for("robot.jpeg")


def test_save_writes_the_format_the_extension_asks_for(tmp_path: Path) -> None:
    png = tmp_path / "robot.png"
    svg = tmp_path / "robot.svg"

    assert save(QUADRUPED, png) == "png"
    assert save(QUADRUPED, svg) == "svg"
    assert png.read_bytes().startswith(PNG_SIGNATURE)
    assert svg.read_text(encoding="utf-8").lstrip().startswith("<?xml")


def test_save_can_be_overridden(tmp_path: Path) -> None:
    output = tmp_path / "robot.svg"
    assert save(QUADRUPED, output, fmt="png") == "png"
    assert output.read_bytes().startswith(PNG_SIGNATURE)


def test_save_rejects_an_unknown_format(tmp_path: Path) -> None:
    with pytest.raises(PreviewError, match="unknown format"):
        save(QUADRUPED, tmp_path / "robot.svg", fmt="tiff")
