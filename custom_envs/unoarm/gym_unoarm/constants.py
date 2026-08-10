from __future__ import annotations

from pathlib import Path

DT = 0.05
FPS = 20

ASSETS_DIR = Path(__file__).parent.resolve()
XML_PATH = ASSETS_DIR / "mujoco_unoarm.xml"

# Three-view observation: a top camera plus a wrist camera on each arm. All
# three cameras are defined in the MuJoCo XML (left_wrist / right_wrist hang on
# Left/Right_Link7, top hangs on the worldbody). Kept as RGB only; no depth.
CAMERAS = ("top", "left_wrist", "right_wrist")
IMG_H = 480
IMG_W = 640

# Park disabled props far away so they leave the RGB views.
PROP_PARK_POS = (80.0, 80.0, 80.0)

# Post-grasp waypoint (sword-handle must pass near this point). World / base frame, meters.
DEFAULT_EXECUTION_POINT_POS = (-0.15, -0.35, 1.15)

ARM_JOINTS = (
    "Left_Joint1",
    "Left_Joint2",
    "Left_Joint3",
    "Left_Joint4",
    "Left_Joint5",
    "Left_Joint6",
    "Left_Joint7",
    "Right_Joint1",
    "Right_Joint2",
    "Right_Joint3",
    "Right_Joint4",
    "Right_Joint5",
    "Right_Joint6",
    "Right_Joint7",
)

CONTROL_JOINTS = (
    "Left_Joint1",
    "Left_Joint2",
    "Left_Joint3",
    "Left_Joint4",
    "Left_Joint5",
    "Left_Joint6",
    "Left_Joint7",
    "Left_Gripper_Joint",
    "Right_Joint1",
    "Right_Joint2",
    "Right_Joint3",
    "Right_Joint4",
    "Right_Joint5",
    "Right_Joint6",
    "Right_Joint7",
    "Right_Gripper_Joint",
)

JOINTS = (
    "Left_Joint1",
    "Left_Joint2",
    "Left_Joint3",
    "Left_Joint4",
    "Left_Joint5",
    "Left_Joint6",
    "Left_Joint7",
    "Left_Gripper",
    "Right_Joint1",
    "Right_Joint2",
    "Right_Joint3",
    "Right_Joint4",
    "Right_Joint5",
    "Right_Joint6",
    "Right_Joint7",
    "Right_Gripper",
)

ACTIONS = JOINTS
START_POSE = (0.0,) * 16

MIMIC_JOINTS = {
    "Left_Gripper_Left_Support_Joint": ("Left_Gripper_Joint", -1.0, 0.0),
    "Left_Gripper_Left_2_Joint": ("Left_Gripper_Joint", 1.0, 0.0),
    "Left_Gripper_Right_2_Joint": ("Left_Gripper_Joint", -1.0, 0.0),
    "Left_Gripper_Right_1_Joint": ("Left_Gripper_Joint", -1.0, 0.0),
    "Left_Gripper_Right_Support_Joint": ("Left_Gripper_Joint", -1.0, 0.0),
    "Right_Gripper_Left_Support_Joint": ("Right_Gripper_Joint", -1.0, 0.0),
    "Right_Gripper_Left_2_Joint": ("Right_Gripper_Joint", 1.0, 0.0),
    "Right_Gripper_Right_2_Joint": ("Right_Gripper_Joint", -1.0, 0.0),
    "Right_Gripper_Right_1_Joint": ("Right_Gripper_Joint", -1.0, 0.0),
    "Right_Gripper_Right_Support_Joint": ("Right_Gripper_Joint", -1.0, 0.0),
}

GRIPPER_OPEN = 1.0
GRIPPER_CLOSE = -1.0
# Normalized gripper joints: -1 ≈ fully closed (raw ≈ -0.91), +1 ≈ open (raw ≈ 0).
GRASP_CLOSE_THRESHOLD = -0.35
GRASP_RELEASE_THRESHOLD = -0.15
LEFT_GRIPPER_INDEX = 7  # CONTROL_JOINTS index
RIGHT_GRIPPER_INDEX = 15  # CONTROL_JOINTS index

# --- Reach-sword scene (kinematic dual targets: left shield + right sword) ---
SCENE_FREE_SPACE = "free_space"
SCENE_REACH_SWORD = "reach_sword"
SCENE_TABLE_PLACE = "table_place"
TASK_REACH_SWORD = "Grasp the shield with the left arm and the sword with the right arm"
TASK_TABLE_PLACE = "Pick up the cylinder and place it on the target circle"
REACH_SUCCESS_THRESHOLD = 0.05

# Mesh is ~1 m along +Z in the STL. Tip near Z≈0; grip near Z≈1.
SWORD_MESH_SCALE = 1.20  # 0.80 × 1.5
# 180° about X: tip (Z≈0) ends above, grip (Z≈1) ends below in world.
SWORD_BODY_QUAT = (0.0, 1.0, 0.0, 0.0)  # wxyz
# Grip site in body/mesh frame (scales with mesh). With Rx(180), world handle is below tip.
SWORD_HANDLE_LOCAL = (0.0, 0.0, 1.11)  # 0.74 × 1.5
# Front of mecha in Web view (+Y); +X is robot left, −X is robot right.
# handle_world_z ≈ body_z - 1.11 ≈ 1.00 → body_z ≈ 2.11
SWORD_BODY_POS = (-0.35, -0.55, 2.11)

# Helmet/shield (left-arm target), opposite side of the sword.
# Raw STL ≈ 0.92 × 0.52 × 1.0 m; scale down for dual-arm reach workspace.
HELMET_MESH_SCALE = 0.45
HELMET_BODY_QUAT = (1.0, 0.0, 0.0, 0.0)  # wxyz upright
# Front face toward robot (−Y), mid-height grasp point in body frame (after mesh scale).
# Moved +0.21 m back (+Y) and +0.05 m up (+Z) from the initial front-face site.
HELMET_HANDLE_LOCAL = (0.0, 0.09, 0.27)
# +X = robot left; handle ≈ body + local → ~1.0 m height.
HELMET_BODY_POS = (0.35, -0.55, 0.78)

# TCP approximation in Link7 frame (along gripper +X).
LEFT_TCP_LOCAL = (0.26, 0.0, 0.0)
RIGHT_TCP_LOCAL = (0.26, 0.0, 0.0)
REACH_TCP_SITE = "right_tcp"  # default metric / validate (right arm → sword)

# Unused by reach scene (targets float); kept for optional reuse.
TABLE_POS = (-0.10, -0.55, 0.72)
TABLE_SIZE = (0.22, 0.22, 0.03)

# --- Table-place scene (table + peg + fixed target circle) ---
# Box geom uses half-sizes. Top Z = TABLE_PLACE_POS[2] + TABLE_PLACE_HALF[2].
# Y is further into the workspace (−Y) so the table sits ~20 cm farther from
# the robot base than the original −0.55 m placement.
TABLE_PLACE_POS = (0.0, -0.75, 0.78)
TABLE_PLACE_HALF = (0.38, 0.28, 0.04)
TABLE_PLACE_RGBA = (0.55, 0.42, 0.28, 1.0)
TABLE_PLACE_THREE_COLOR = 0x8c6b47
# Peg: upright cylinder; body at geometric center; grasp site at top.
# Radius sized so the right-gripper main fingers (inner spacing ~0.046 m) can
# wrap around it without clipping through the mesh.
PEG_RADIUS = 0.018
PEG_HALF_HEIGHT = 0.045  # full height 9 cm
PEG_RGBA = (0.85, 0.35, 0.2, 1.0)
PEG_THREE_COLOR = 0xd95a33
# Default peg XY on the table (robot-right / −X). Z recomputed from table top.
# Keep near the robot-facing table edge so IK stays reachable after the table
# was shifted −0.20 m in Y.
PEG_DEFAULT_XY = (-0.20, -0.58)
# Fixed target circle on table surface (world XY); Z = table top.
TABLE_CIRCLE_CENTER_XY = (0.0, -0.60)
TABLE_CIRCLE_RADIUS = 0.06
TABLE_CIRCLE_RGBA = (0.15, 0.75, 0.35, 0.9)
TABLE_CIRCLE_THREE_COLOR = 0x26bf59

# Multi-point peg sampling: stay on the table AND inside a reachable workspace.
# Bounds are world XY (m). Generation intersects table-inset ∩ this box.
PEG_SAMPLE_EDGE_MARGIN = 0.05  # inset from table rim (plus PEG_RADIUS)
PEG_WORKSPACE_X_RANGE = (-0.30, 0.08)
PEG_WORKSPACE_Y_RANGE = (-0.70, -0.50)
# Reject samples already inside/near the place circle (task would be trivial).
PEG_MIN_DIST_FROM_CIRCLE = float(TABLE_CIRCLE_RADIUS + PEG_RADIUS + 0.03)

# Peg free-fall physics (env-layer vertical integration on release).
# The peg is a kinematic body driven by direct body_pos writes; on release we
# integrate a simple vertical free-fall (gravity + table contact) instead of
# snapping it to the table instantly.
PEG_GRAVITY = 9.81          # m/s^2, downward
PEG_FALL_MAX_STEPS = 60     # safety: stop falling after this many env steps
PEG_REST_Z_OFFSET = 0.0001  # tiny lift above table to avoid z-fighting

# Grasp detection for table_place: the peg is grasped when BOTH jaw sides are
# nearly touching the peg cylinder (MuJoCo signed geom distance
# <= PEG_GRASP_SURFACE_TOL; negative = penetration / clamp), AND the peg center
# lies between the two finger-mesh frames along the jaw opening axis.
# Distances use finger mesh geoms — not body origins at the hinges.
PEG_GRASP_SURFACE_TOL = 0.015  # max signed finger-mesh → peg gap (m); ≤0 = touching
PEG_GRASP_DIST_MAX = 1.0  # mj_geomDistance search bound (m)
# Peg projection t along left→right finger segment must lie in this open interval
# (0 = at left pad frame, 1 = at right). Rejects same-side poke-through.
PEG_GRASP_BETWEEN_LO = 0.05
PEG_GRASP_BETWEEN_HI = 0.95

# Visual: bright white (MuJoCo rgba + Three.js hex must stay in sync).
SWORD_RGBA = (0.96, 0.97, 0.99, 1.0)
SWORD_THREE_COLOR = 0xf5f7fa
HELMET_RGBA = (0.72, 0.78, 0.86, 1.0)
HELMET_THREE_COLOR = 0xb8c6db

REACH_SWORD_XML_PATH = ASSETS_DIR / "mujoco_unoarm_reach_sword.xml"
TABLE_PLACE_XML_PATH = ASSETS_DIR / "mujoco_unoarm_table_place.xml"
SWORD_MESH_PATH = ASSETS_DIR / "meshes" / "sword.stl"
HELMET_MESH_PATH = ASSETS_DIR / "meshes" / "helmet.stl"


def table_place_top_z() -> float:
    """World Z of the table top surface."""
    return float(TABLE_PLACE_POS[2] + TABLE_PLACE_HALF[2])


def peg_body_pos_from_xy(xy: tuple[float, float] | list[float]) -> tuple[float, float, float]:
    """Peg body center so the cylinder rests on the table top."""
    x, y = float(xy[0]), float(xy[1])
    return (x, y, table_place_top_z() + float(PEG_HALF_HEIGHT))


def peg_table_inset_xy_bounds(
    edge_margin: float = PEG_SAMPLE_EDGE_MARGIN,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Axis-aligned peg XY bounds inset from the table rim (includes peg radius)."""
    m = float(edge_margin) + float(PEG_RADIUS)
    cx, cy = float(TABLE_PLACE_POS[0]), float(TABLE_PLACE_POS[1])
    hx, hy = float(TABLE_PLACE_HALF[0]), float(TABLE_PLACE_HALF[1])
    lo = (cx - hx + m, cy - hy + m)
    hi = (cx + hx - m, cy + hy - m)
    return lo, hi


def peg_workspace_xy_bounds(
    *,
    edge_margin: float = PEG_SAMPLE_EDGE_MARGIN,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Reachable peg sampling box = table inset ∩ configured workspace ranges."""
    (tx0, ty0), (tx1, ty1) = peg_table_inset_xy_bounds(edge_margin)
    xr = x_range if x_range is not None else PEG_WORKSPACE_X_RANGE
    yr = y_range if y_range is not None else PEG_WORKSPACE_Y_RANGE
    x0 = max(float(tx0), float(min(xr[0], xr[1])))
    x1 = min(float(tx1), float(max(xr[0], xr[1])))
    y0 = max(float(ty0), float(min(yr[0], yr[1])))
    y1 = min(float(ty1), float(max(yr[0], yr[1])))
    if x1 < x0 or y1 < y0:
        raise ValueError(
            f"Empty peg workspace after intersecting table inset with "
            f"x={xr}, y={yr}, margin={edge_margin}"
        )
    return (x0, y0), (x1, y1)


def is_peg_xy_in_workspace(
    xy: tuple[float, float] | list[float],
    *,
    edge_margin: float = PEG_SAMPLE_EDGE_MARGIN,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    min_dist_from_circle: float = PEG_MIN_DIST_FROM_CIRCLE,
) -> bool:
    """True when peg XY is on-table, in workspace, and clear of the place circle."""
    x, y = float(xy[0]), float(xy[1])
    (x0, y0), (x1, y1) = peg_workspace_xy_bounds(
        edge_margin=edge_margin, x_range=x_range, y_range=y_range
    )
    if not (x0 - 1e-9 <= x <= x1 + 1e-9 and y0 - 1e-9 <= y <= y1 + 1e-9):
        return False
    cx, cy = float(TABLE_CIRCLE_CENTER_XY[0]), float(TABLE_CIRCLE_CENTER_XY[1])
    dist = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
    return dist >= float(min_dist_from_circle) - 1e-9


def clip_peg_xy_to_table(
    xy: tuple[float, float] | list[float],
    *,
    edge_margin: float = PEG_SAMPLE_EDGE_MARGIN,
) -> tuple[float, float]:
    """Clamp peg XY so the cylinder stays fully on the table top."""
    x, y = float(xy[0]), float(xy[1])
    (x0, y0), (x1, y1) = peg_table_inset_xy_bounds(edge_margin)
    return (min(max(x, x0), x1), min(max(y, y0), y1))


def clip_peg_xy_to_workspace(
    xy: tuple[float, float] | list[float],
    *,
    edge_margin: float = PEG_SAMPLE_EDGE_MARGIN,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """Clamp peg XY into the reachable on-table workspace box."""
    x, y = float(xy[0]), float(xy[1])
    (x0, y0), (x1, y1) = peg_workspace_xy_bounds(
        edge_margin=edge_margin, x_range=x_range, y_range=y_range
    )
    return (min(max(x, x0), x1), min(max(y, y0), y1))


def sample_peg_xy(
    rng,
    *,
    edge_margin: float = PEG_SAMPLE_EDGE_MARGIN,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    min_dist_from_circle: float = PEG_MIN_DIST_FROM_CIRCLE,
    max_tries: int = 200,
) -> tuple[float, float]:
    """Sample a peg XY inside the workspace bounds (never outside the table)."""
    (x0, y0), (x1, y1) = peg_workspace_xy_bounds(
        edge_margin=edge_margin, x_range=x_range, y_range=y_range
    )
    for _ in range(int(max_tries)):
        xy = (float(rng.uniform(x0, x1)), float(rng.uniform(y0, y1)))
        if is_peg_xy_in_workspace(
            xy,
            edge_margin=edge_margin,
            x_range=x_range,
            y_range=y_range,
            min_dist_from_circle=min_dist_from_circle,
        ):
            return xy
    raise RuntimeError(
        f"Failed to sample peg XY in workspace after {max_tries} tries "
        f"(box=[{x0:.3f},{x1:.3f}]x[{y0:.3f},{y1:.3f}], "
        f"min_circle_dist={min_dist_from_circle})"
    )
