"""Sword target pose helpers: handle world XYZ + yaw/pitch/roll → body pose."""

from __future__ import annotations

import math

import numpy as np

from .constants import (
    HELMET_BODY_POS,
    HELMET_BODY_QUAT,
    HELMET_HANDLE_LOCAL,
    SWORD_BODY_POS,
    SWORD_BODY_QUAT,
    SWORD_HANDLE_LOCAL,
)


def quat_wxyz_normalize(q: tuple[float, float, float, float] | np.ndarray) -> np.ndarray:
    qn = np.asarray(q, dtype=np.float64).reshape(4)
    n = float(np.linalg.norm(qn))
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return qn / n


def quat_wxyz_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product, both wxyz."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def quat_wxyz_from_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    ax = np.asarray(axis, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(ax))
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    ax = ax / n
    half = 0.5 * float(angle_rad)
    s = math.sin(half)
    return np.array([math.cos(half), ax[0] * s, ax[1] * s, ax[2] * s], dtype=np.float64)


def quat_wxyz_to_rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_wxyz_normalize(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def euler_zyx_deg_to_quat(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Extrinsic ZYX (yaw about Z, pitch about Y, roll about X), degrees → wxyz."""
    qz = quat_wxyz_from_axis_angle(np.array([0.0, 0.0, 1.0]), math.radians(yaw_deg))
    qy = quat_wxyz_from_axis_angle(np.array([0.0, 1.0, 0.0]), math.radians(pitch_deg))
    qx = quat_wxyz_from_axis_angle(np.array([1.0, 0.0, 0.0]), math.radians(roll_deg))
    return quat_wxyz_normalize(quat_wxyz_mul(qz, quat_wxyz_mul(qy, qx)))


def sword_quat_from_euler_deg(
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
    *,
    base_quat: tuple[float, float, float, float] = SWORD_BODY_QUAT,
) -> tuple[float, float, float, float]:
    """User euler applied on top of upright base (default Rx 180°)."""
    user = euler_zyx_deg_to_quat(yaw_deg, pitch_deg, roll_deg)
    base = quat_wxyz_normalize(base_quat)
    q = quat_wxyz_mul(user, base)
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))


def body_pos_from_handle_world(
    handle_world: tuple[float, float, float] | np.ndarray,
    body_quat_wxyz: tuple[float, float, float, float] | np.ndarray,
    *,
    handle_local: tuple[float, float, float] = SWORD_HANDLE_LOCAL,
) -> tuple[float, float, float]:
    """body_pos = handle_world - R(quat) @ handle_local."""
    hw = np.asarray(handle_world, dtype=np.float64).reshape(3)
    hl = np.asarray(handle_local, dtype=np.float64).reshape(3)
    r = quat_wxyz_to_rotmat(np.asarray(body_quat_wxyz, dtype=np.float64))
    bp = hw - r @ hl
    return (float(bp[0]), float(bp[1]), float(bp[2]))


def handle_world_from_body(
    body_pos: tuple[float, float, float] | np.ndarray = SWORD_BODY_POS,
    body_quat_wxyz: tuple[float, float, float, float] | np.ndarray = SWORD_BODY_QUAT,
    *,
    handle_local: tuple[float, float, float] = SWORD_HANDLE_LOCAL,
) -> tuple[float, float, float]:
    bp = np.asarray(body_pos, dtype=np.float64).reshape(3)
    hl = np.asarray(handle_local, dtype=np.float64).reshape(3)
    r = quat_wxyz_to_rotmat(np.asarray(body_quat_wxyz, dtype=np.float64))
    hw = bp + r @ hl
    return (float(hw[0]), float(hw[1]), float(hw[2]))


def resolve_sword_pose(
    *,
    handle_pos: tuple[float, float, float] | list[float] | None = None,
    euler_deg: tuple[float, float, float] | list[float] | None = None,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float], tuple[float, float, float]]:
    """Return (body_pos, body_quat_wxyz, handle_world) from UI settings."""
    yaw, pitch, roll = (0.0, 0.0, 0.0) if euler_deg is None else tuple(float(x) for x in euler_deg)
    quat = sword_quat_from_euler_deg(yaw, pitch, roll)
    if handle_pos is None:
        handle = handle_world_from_body(SWORD_BODY_POS, SWORD_BODY_QUAT)
    else:
        handle = tuple(float(x) for x in handle_pos)  # type: ignore[assignment]
        if len(handle) != 3:
            handle = handle_world_from_body(SWORD_BODY_POS, SWORD_BODY_QUAT)
    body = body_pos_from_handle_world(handle, quat)
    return body, quat, handle  # type: ignore[return-value]


def default_sword_handle_pos() -> list[float]:
    return list(handle_world_from_body())


def default_sword_euler_deg() -> list[float]:
    return [0.0, 0.0, 0.0]


def shield_quat_from_euler_deg(
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
    *,
    base_quat: tuple[float, float, float, float] = HELMET_BODY_QUAT,
) -> tuple[float, float, float, float]:
    """User euler applied on top of upright shield/helmet base."""
    user = euler_zyx_deg_to_quat(yaw_deg, pitch_deg, roll_deg)
    base = quat_wxyz_normalize(base_quat)
    q = quat_wxyz_mul(user, base)
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))


def resolve_shield_pose(
    *,
    handle_pos: tuple[float, float, float] | list[float] | None = None,
    euler_deg: tuple[float, float, float] | list[float] | None = None,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float], tuple[float, float, float]]:
    """Return (body_pos, body_quat_wxyz, handle_world) for the left-arm shield."""
    yaw, pitch, roll = (0.0, 0.0, 0.0) if euler_deg is None else tuple(float(x) for x in euler_deg)
    quat = shield_quat_from_euler_deg(yaw, pitch, roll)
    if handle_pos is None:
        handle = handle_world_from_body(HELMET_BODY_POS, HELMET_BODY_QUAT, handle_local=HELMET_HANDLE_LOCAL)
    else:
        handle = tuple(float(x) for x in handle_pos)  # type: ignore[assignment]
        if len(handle) != 3:
            handle = handle_world_from_body(
                HELMET_BODY_POS, HELMET_BODY_QUAT, handle_local=HELMET_HANDLE_LOCAL
            )
    body = body_pos_from_handle_world(handle, quat, handle_local=HELMET_HANDLE_LOCAL)
    return body, quat, handle  # type: ignore[return-value]


def default_shield_handle_pos() -> list[float]:
    return list(handle_world_from_body(HELMET_BODY_POS, HELMET_BODY_QUAT, handle_local=HELMET_HANDLE_LOCAL))


def default_shield_euler_deg() -> list[float]:
    return [0.0, 0.0, 0.0]
