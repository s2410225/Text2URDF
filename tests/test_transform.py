"""Homogeneous transform, RPY and mirroring tests (phase 1)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from urdfgen.builder.inertia import (
    InertiaError,
    box_inertia,
    cylinder_inertia,
    primitive_inertia,
    primitive_inertial,
    sphere_inertia,
)
from urdfgen.builder.transform import (
    MirrorAxis,
    compose,
    identity,
    inertia_to_matrix,
    invert,
    make_transform,
    matrix_to_inertia,
    matrix_to_origin,
    matrix_to_rpy,
    mirror_axis_vector,
    mirror_inertia,
    mirror_limits,
    mirror_matrix,
    mirror_origin,
    pose_to_matrix,
    rotate_inertia,
    rpy_to_matrix,
    transform_point,
)
from urdfgen.catalog.schema import BoxGeometry, Inertia, MeshGeometry, Origin

HALF_PI = math.pi / 2.0

RPY_SAMPLES = [
    (0.0, 0.0, 0.0),
    (0.3, 0.0, 0.0),
    (0.0, -0.4, 0.0),
    (0.0, 0.0, 1.2),
    (0.3, -0.4, 1.2),
    (-2.9, 1.1, 2.7),
    (0.1, 1.4, -0.8),
]


# --- rotation -------------------------------------------------------------


@pytest.mark.parametrize("rpy", RPY_SAMPLES)
def test_rpy_matrix_roundtrip(rpy: tuple[float, float, float]) -> None:
    recovered = matrix_to_rpy(rpy_to_matrix(rpy))
    np.testing.assert_allclose(rpy_to_matrix(recovered), rpy_to_matrix(rpy), atol=1e-12)


def test_rpy_is_fixed_axis_zyx() -> None:
    """URDF RPY is ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``, not the reverse order."""
    roll, pitch, yaw = 0.3, -0.4, 1.2
    expected = _rot_z(yaw) @ _rot_y(pitch) @ _rot_x(roll)
    np.testing.assert_allclose(rpy_to_matrix((roll, pitch, yaw)), expected, atol=1e-12)


def test_quarter_turn_about_z() -> None:
    rotation = rpy_to_matrix((0.0, 0.0, HALF_PI))
    np.testing.assert_allclose(rotation @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-12)


@pytest.mark.parametrize("pitch", [HALF_PI, -HALF_PI])
@pytest.mark.parametrize("roll", [0.0, 0.7, -1.9])
def test_gimbal_lock_roundtrips(pitch: float, roll: float) -> None:
    """At pitch = +-pi/2 roll and yaw degenerate; the matrix must still match."""
    original = rpy_to_matrix((roll, pitch, 0.0))
    recovered = matrix_to_rpy(original)
    assert recovered[2] == 0.0
    assert math.isclose(abs(recovered[1]), HALF_PI, abs_tol=1e-12)
    np.testing.assert_allclose(rpy_to_matrix(recovered), original, atol=1e-12)


def test_gimbal_lock_absorbs_yaw_into_roll() -> None:
    """A locked pose given as roll+yaw comes back as a single roll."""
    original = rpy_to_matrix((0.4, HALF_PI, 0.9))
    recovered = matrix_to_rpy(original)
    np.testing.assert_allclose(rpy_to_matrix(recovered), original, atol=1e-12)


def test_matrix_to_rpy_clamps_rounding_overshoot() -> None:
    """A matrix whose r20 is slightly beyond -1 must not produce a domain error."""
    rotation = rpy_to_matrix((0.0, HALF_PI, 0.0))
    rotation[2, 0] = -1.0 - 1e-15
    assert math.isclose(matrix_to_rpy(rotation)[1], HALF_PI, abs_tol=1e-9)


# --- composition ----------------------------------------------------------


def test_compose_of_nothing_is_identity() -> None:
    np.testing.assert_allclose(compose(), identity())


def test_compose_is_associative() -> None:
    a = make_transform((1.0, 2.0, 3.0), (0.1, 0.2, 0.3))
    b = make_transform((-0.5, 0.0, 2.0), (0.0, -1.1, 0.4))
    c = make_transform((0.0, 4.0, -1.0), (0.9, 0.0, 0.0))
    np.testing.assert_allclose(compose(compose(a, b), c), compose(a, compose(b, c)), atol=1e-12)


def test_invert_is_a_true_inverse() -> None:
    transform = make_transform((1.0, -2.0, 0.5), (0.4, 1.1, -0.7))
    np.testing.assert_allclose(compose(transform, invert(transform)), identity(), atol=1e-12)


def test_translation_then_rotation_is_not_commutative() -> None:
    """The reason poses compose as matrices instead of by adding angles."""
    translate = make_transform((1.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    rotate = make_transform((0.0, 0.0, 0.0), (0.0, 0.0, HALF_PI))
    np.testing.assert_allclose(
        transform_point(compose(rotate, translate), (0, 0, 0)), (0, 1, 0), atol=1e-12
    )
    np.testing.assert_allclose(
        transform_point(compose(translate, rotate), (0, 0, 0)), (1, 0, 0), atol=1e-12
    )


def test_compose_matches_hand_computed_pose() -> None:
    """Rotate a quarter turn about Z, then step 2 m along the rotated X axis."""
    result = compose(
        make_transform((0.0, 0.0, 1.0), (0.0, 0.0, HALF_PI)),
        make_transform((2.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    )
    origin = matrix_to_origin(result)
    np.testing.assert_allclose(origin.xyz, (0.0, 2.0, 1.0), atol=1e-12)
    np.testing.assert_allclose(origin.rpy, (0.0, 0.0, HALF_PI), atol=1e-12)


def test_pose_to_matrix_accepts_origin_and_pose() -> None:
    from urdfgen.ir.assembly import Pose

    origin = Origin(xyz=(1.0, 2.0, 3.0), rpy=(0.1, 0.2, 0.3))
    pose = Pose(xyz=(1.0, 2.0, 3.0), rpy=(0.1, 0.2, 0.3))
    np.testing.assert_allclose(pose_to_matrix(origin), pose_to_matrix(pose))


# --- inertia tensors ------------------------------------------------------


def test_inertia_matrix_roundtrip() -> None:
    inertia = Inertia(ixx=1.0, iyy=2.0, izz=3.0, ixy=0.1, ixz=-0.2, iyz=0.3)
    assert matrix_to_inertia(inertia_to_matrix(inertia)) == inertia


def test_rotate_inertia_preserves_trace() -> None:
    """``R I R^T`` is a similarity transform, so the trace is invariant."""
    inertia = Inertia(ixx=1.0, iyy=2.0, izz=3.0)
    rotated = rotate_inertia(inertia, rpy_to_matrix((0.3, -0.4, 1.2)))
    assert math.isclose(rotated.ixx + rotated.iyy + rotated.izz, 6.0, abs_tol=1e-12)


def test_box_inertia_matches_appendix_a() -> None:
    inertia = box_inertia(12.0, (1.0, 2.0, 3.0))
    assert math.isclose(inertia.ixx, 12.0 * (4.0 + 9.0) / 12.0)
    assert math.isclose(inertia.iyy, 12.0 * (1.0 + 9.0) / 12.0)
    assert math.isclose(inertia.izz, 12.0 * (1.0 + 4.0) / 12.0)


def test_cylinder_inertia_matches_appendix_a() -> None:
    inertia = cylinder_inertia(2.0, 0.5, 3.0)
    assert math.isclose(inertia.ixx, 2.0 * (3 * 0.25 + 9.0) / 12.0)
    assert math.isclose(inertia.ixx, inertia.iyy)
    assert math.isclose(inertia.izz, 2.0 * 0.25 / 2.0)


def test_sphere_inertia_matches_appendix_a() -> None:
    inertia = sphere_inertia(5.0, 0.2)
    assert math.isclose(inertia.ixx, 2.0 * 5.0 * 0.04 / 5.0)
    assert inertia.ixx == inertia.iyy == inertia.izz


def test_primitive_inertia_rejects_meshes() -> None:
    with pytest.raises(InertiaError, match="mesh-based links"):
        primitive_inertia(1.0, MeshGeometry(mesh="meshes/x.stl"))


def test_primitive_inertia_rejects_non_positive_mass() -> None:
    with pytest.raises(InertiaError, match="mass must be positive"):
        primitive_inertia(0.0, BoxGeometry(size=(1.0, 1.0, 1.0)))


def test_rotated_primitive_gains_off_diagonal_terms() -> None:
    """Appendix A: a rotated origin means ``I' = R I R^T``, not a diagonal tensor."""
    inertial = primitive_inertial(
        1.0,
        BoxGeometry(size=(1.0, 2.0, 3.0)),
        Origin(rpy=(0.0, 0.0, 0.6)),
    )
    assert inertial.inertia is not None
    assert abs(inertial.inertia.ixy) > 1e-6


# --- mirroring ------------------------------------------------------------


def test_mirror_matrix_is_its_own_inverse() -> None:
    reflection = mirror_matrix("y")
    np.testing.assert_allclose(reflection @ reflection, identity())


def test_mirror_none_is_the_identity() -> None:
    origin = Origin(xyz=(1.0, 2.0, 3.0), rpy=(0.1, 0.2, 0.3))
    assert mirror_origin(origin, "none") == origin
    assert mirror_axis_vector((0.0, 1.0, 0.0), "none") == (0.0, 1.0, 0.0)
    assert mirror_limits(-1.0, 2.0, "none") == (-1.0, 2.0)


def test_mirror_origin_flips_only_the_mirror_axis() -> None:
    mirrored = mirror_origin(Origin(xyz=(1.0, 2.0, 3.0)), "y")
    np.testing.assert_allclose(mirrored.xyz, (1.0, -2.0, 3.0))


def test_mirror_origin_conjugates_the_rotation() -> None:
    """``M R M`` stays a proper rotation, so the result is still expressible as RPY."""
    original = Origin(rpy=(0.3, -0.4, 1.2))
    mirrored = mirror_origin(original, "x")
    expected = mirror_matrix("x")[:3, :3] @ rpy_to_matrix(original.rpy) @ mirror_matrix("x")[:3, :3]
    np.testing.assert_allclose(rpy_to_matrix(mirrored.rpy), expected, atol=1e-12)
    assert math.isclose(float(np.linalg.det(rpy_to_matrix(mirrored.rpy))), 1.0, abs_tol=1e-12)


def test_mirror_inertia_flips_products_containing_the_axis_once() -> None:
    """Spec 5.4: an X mirror negates ixy and ixz but leaves iyz alone."""
    inertia = Inertia(ixx=1.0, iyy=2.0, izz=3.0, ixy=0.1, ixz=0.2, iyz=0.3)
    mirrored = mirror_inertia(inertia, "x")
    assert (mirrored.ixx, mirrored.iyy, mirrored.izz) == (1.0, 2.0, 3.0)
    assert math.isclose(mirrored.ixy, -0.1)
    assert math.isclose(mirrored.ixz, -0.2)
    assert math.isclose(mirrored.iyz, 0.3)


def test_mirror_axis_negates_the_mirror_component() -> None:
    assert mirror_axis_vector((1.0, 1.0, 1.0), "x") == (-1.0, 1.0, 1.0)
    assert mirror_axis_vector((1.0, 1.0, 1.0), "y") == (1.0, -1.0, 1.0)


def test_mirror_limits_swap_and_negate() -> None:
    assert mirror_limits(-2.36, 0.0, "y") == (0.0, 2.36)


@pytest.mark.parametrize("axis", ["x", "y"])
@pytest.mark.parametrize("angle", [0.0, 0.5, -1.3])
def test_mirrored_joint_reproduces_the_mirrored_rotation(axis: MirrorAxis, angle: float) -> None:
    """``M R(a, q) M == R(M a, -q)``.

    This is the identity that makes the axis rule and the limit rule correct
    only as a pair: the axis is reflected, and the joint variable flips sign.
    """
    joint_axis = (0.3, 0.9, -0.4)
    reflection = mirror_matrix(axis)[:3, :3]
    mirrored_rotation = reflection @ _rotation_about(joint_axis, angle) @ reflection
    expected = _rotation_about(mirror_axis_vector(joint_axis, axis), -angle)
    np.testing.assert_allclose(mirrored_rotation, expected, atol=1e-12)


# --- helpers --------------------------------------------------------------


def _rot_x(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _rotation_about(axis: tuple[float, float, float], angle: float) -> np.ndarray:
    """Rodrigues' formula, used to check the mirroring identity independently."""
    unit = np.asarray(axis, dtype=np.float64)
    unit = unit / np.linalg.norm(unit)
    skew = np.array(
        [[0, -unit[2], unit[1]], [unit[2], 0, -unit[0]], [-unit[1], unit[0], 0]],
        dtype=np.float64,
    )
    return np.eye(3) + math.sin(angle) * skew + (1 - math.cos(angle)) * (skew @ skew)
