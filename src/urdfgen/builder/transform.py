"""Homogeneous transforms, RPY conversion and mirroring.

All pose composition in this project goes through 4x4 homogeneous matrices.
Adding or subtracting Euler angles is wrong and must not appear anywhere in the
codebase (spec section 6.2 and 12).

RPY follows the URDF convention: fixed-axis Z-Y-X, i.e. roll about X, then
pitch about Y, then yaw about Z, giving ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import numpy.typing as npt

from urdfgen.catalog.schema import Inertia, Origin
from urdfgen.ir.assembly import Pose

Vec3 = tuple[float, float, float]
Matrix = npt.NDArray[np.float64]

MirrorAxis = Literal["none", "x", "y", "z"]

_AXIS_INDEX: dict[str, int] = {"x": 0, "y": 1, "z": 2}

GIMBAL_LOCK_TOLERANCE = 1e-9
"""``|sin(pitch)|`` within this distance of 1 is treated as gimbal lock."""


def identity() -> Matrix:
    return np.eye(4, dtype=np.float64)


def rpy_to_matrix(rpy: Vec3) -> Matrix:
    """Rotation matrix for fixed-axis Z-Y-X roll/pitch/yaw, in radians."""
    roll, pitch, yaw = (float(v) for v in rpy)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def matrix_to_rpy(rotation: Matrix) -> Vec3:
    """Invert :func:`rpy_to_matrix`, handling gimbal lock explicitly.

    At ``pitch = +-pi/2`` roll and yaw describe the same rotation, so their
    split is arbitrary. We resolve it by fixing ``yaw = 0`` and putting the
    whole remaining rotation into roll, which keeps the result deterministic
    and round-trips through :func:`rpy_to_matrix`.
    """
    rot = np.asarray(rotation, dtype=np.float64)[:3, :3]

    sin_pitch = -float(rot[2, 0])
    sin_pitch = min(1.0, max(-1.0, sin_pitch))

    if abs(sin_pitch) >= 1.0 - GIMBAL_LOCK_TOLERANCE:
        pitch = math.copysign(math.pi / 2.0, sin_pitch)
        yaw = 0.0
        # With yaw = 0 and cos(pitch) = 0 the matrix reduces to
        #   r01 = -sin(pitch) * sin(roll) * ... -> use the r01/r11 pair below.
        roll = math.atan2(sin_pitch * float(rot[0, 1]), float(rot[1, 1]))
        return (roll, pitch, yaw)

    pitch = math.asin(sin_pitch)
    roll = math.atan2(float(rot[2, 1]), float(rot[2, 2]))
    yaw = math.atan2(float(rot[1, 0]), float(rot[0, 0]))
    return (roll, pitch, yaw)


def make_transform(xyz: Vec3, rpy: Vec3) -> Matrix:
    """4x4 homogeneous transform from a translation and an RPY triple."""
    matrix = identity()
    matrix[:3, :3] = rpy_to_matrix(rpy)
    matrix[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return matrix


def pose_to_matrix(pose: Pose | Origin) -> Matrix:
    return make_transform(pose.xyz, pose.rpy)


def matrix_to_origin(matrix: Matrix) -> Origin:
    xyz, rpy = _decompose(matrix)
    return Origin(xyz=xyz, rpy=rpy)


def matrix_to_pose(matrix: Matrix) -> Pose:
    xyz, rpy = _decompose(matrix)
    return Pose(xyz=xyz, rpy=rpy)


def invert(matrix: Matrix) -> Matrix:
    """Inverse of a rigid transform, using ``R^T`` rather than a general solve."""
    rot = np.asarray(matrix, dtype=np.float64)[:3, :3]
    translation = np.asarray(matrix, dtype=np.float64)[:3, 3]
    result = identity()
    result[:3, :3] = rot.T
    result[:3, 3] = -rot.T @ translation
    return result


def compose(*matrices: Matrix) -> Matrix:
    """Left-to-right product of transforms; ``compose()`` is the identity."""
    result = identity()
    for matrix in matrices:
        result = result @ np.asarray(matrix, dtype=np.float64)
    return result


def transform_direction(matrix: Matrix, direction: Vec3) -> Vec3:
    """Rotate a direction vector (translation is ignored)."""
    rotated = np.asarray(matrix, dtype=np.float64)[:3, :3] @ np.asarray(direction, dtype=np.float64)
    return _as_vec3(rotated)


def transform_point(matrix: Matrix, point: Vec3) -> Vec3:
    homogeneous = np.append(np.asarray(point, dtype=np.float64), 1.0)
    return _as_vec3((np.asarray(matrix, dtype=np.float64) @ homogeneous)[:3])


def inertia_to_matrix(inertia: Inertia) -> Matrix:
    """The symmetric 3x3 inertia tensor."""
    return np.array(
        [
            [inertia.ixx, inertia.ixy, inertia.ixz],
            [inertia.ixy, inertia.iyy, inertia.iyz],
            [inertia.ixz, inertia.iyz, inertia.izz],
        ],
        dtype=np.float64,
    )


def matrix_to_inertia(tensor: Matrix) -> Inertia:
    """Read an :class:`Inertia` back out of a symmetric 3x3 tensor."""
    matrix = np.asarray(tensor, dtype=np.float64)
    return Inertia(
        ixx=float(matrix[0, 0]),
        iyy=float(matrix[1, 1]),
        izz=float(matrix[2, 2]),
        ixy=float(matrix[0, 1]),
        ixz=float(matrix[0, 2]),
        iyz=float(matrix[1, 2]),
    )


def rotate_inertia(inertia: Inertia, rotation: Matrix) -> Inertia:
    """Apply ``I' = R I R^T`` (spec appendix A)."""
    rot = np.asarray(rotation, dtype=np.float64)[:3, :3]
    return matrix_to_inertia(rot @ inertia_to_matrix(inertia) @ rot.T)


# --- Mirroring (spec section 5.4) ----------------------------------------
#
# Mirroring a module across a plane is a reflection, not a rotation, so it
# cannot be expressed as a homogeneous transform without a negative
# determinant. Each affected quantity therefore has its own rule.


def mirror_matrix(axis: MirrorAxis) -> Matrix:
    """The 4x4 reflection matrix for ``axis`` (identity when ``axis`` is none)."""
    matrix = identity()
    if axis != "none":
        matrix[_AXIS_INDEX[axis], _AXIS_INDEX[axis]] = -1.0
    return matrix


def mirror_origin(origin: Origin, axis: MirrorAxis) -> Origin:
    """Reflect a pose: flip the position component and conjugate the rotation.

    The rotation is reflected as ``M R M`` (``M`` is its own inverse), which
    keeps the result a proper rotation and so remains expressible as RPY.
    """
    if axis == "none":
        return origin
    reflection = mirror_matrix(axis)[:3, :3]
    xyz = _as_vec3(reflection @ np.asarray(origin.xyz, dtype=np.float64))
    rotation = reflection @ rpy_to_matrix(origin.rpy) @ reflection
    return Origin(xyz=xyz, rpy=matrix_to_rpy(rotation))


def mirror_inertia(inertia: Inertia, axis: MirrorAxis) -> Inertia:
    """Flip the sign of products of inertia that contain the mirror axis once.

    Mirroring about X negates ``ixy`` and ``ixz`` but leaves ``iyz`` alone,
    because ``iyz`` does not involve the X axis at all.
    """
    if axis == "none":
        return inertia
    reflection = mirror_matrix(axis)[:3, :3]
    return matrix_to_inertia(reflection @ inertia_to_matrix(inertia) @ reflection)


def mirror_axis_vector(vector: Vec3, axis: MirrorAxis) -> Vec3:
    """Negate the component of a joint axis along the mirror axis (``a -> M a``).

    A joint axis is a pseudo-vector, so ``M a`` alone does not reproduce the
    mirrored motion: the joint variable is negated as well. That negation is
    what :func:`mirror_limits` encodes, and the two rules are only correct
    together.

    Concretely, ``M R(a, q) M = R(M a, -q)``: the mirrored joint uses axis
    ``M a`` and runs over the negated range.
    """
    if axis == "none":
        return vector
    reflection = mirror_matrix(axis)[:3, :3]
    return _as_vec3(reflection @ np.asarray(vector, dtype=np.float64))


def mirror_limits(lower: float, upper: float, axis: MirrorAxis) -> tuple[float, float]:
    """Swap and negate joint limits to match the negated joint variable.

    See :func:`mirror_axis_vector` for why the joint variable flips sign.
    """
    if axis == "none":
        return (lower, upper)
    return (-upper, -lower)


def _decompose(matrix: Matrix) -> tuple[Vec3, Vec3]:
    array = np.asarray(matrix, dtype=np.float64)
    return _as_vec3(array[:3, 3]), matrix_to_rpy(array)


def _as_vec3(values: npt.NDArray[np.float64]) -> Vec3:
    return (float(values[0]), float(values[1]), float(values[2]))
