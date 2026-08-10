"""MuJoCo damped-least-squares IK for Unoarm right TCP."""

from __future__ import annotations

import mujoco
import numpy as np

# Gripper axis (+X of right_tcp) points into workspace (−Y).
_x = np.array([0.0, -1.0, 0.0], dtype=np.float64)
_z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
_y = np.cross(_z, _x)
_y = _y / (np.linalg.norm(_y) + 1e-12)
_z = np.cross(_x, _y)
_z = _z / (np.linalg.norm(_z) + 1e-12)
FIXED_APPROACH_ROT_MAT = np.column_stack([_x, _y, _z])

_RIGHT_ARM = tuple(f"Right_Joint{i}" for i in range(1, 8))


def _rot_err(r_cur: np.ndarray, r_tgt: np.ndarray) -> np.ndarray:
    """3-vector orientation error from relative rotation."""
    r_err = r_cur.T @ r_tgt
    return 0.5 * np.array(
        [
            r_err[2, 1] - r_err[1, 2],
            r_err[0, 2] - r_err[2, 0],
            r_err[1, 0] - r_err[0, 1],
        ],
        dtype=np.float64,
    )


def solve_right_tcp_ik(
    env,
    target_pos: np.ndarray,
    *,
    target_rot: np.ndarray | None = None,
    max_iters: int = 300,
    pos_tol: float = 0.01,
    ori_tol: float = 0.1,
    damping: float = 5e-2,
) -> np.ndarray | None:
    """Solve right-arm IK so ``right_tcp`` matches target pose.

    Returns clipped raw 16-D control qpos, or ``None`` on failure.
    Left arm and grippers keep their values from the env state at call time.

    Orientation default: gripper +X along −Y (into workspace), +Z up.
    """
    if int(getattr(env, "_tcp_site_id", -1)) < 0:
        raise RuntimeError("env has no right_tcp site (need reach_sword scene)")
    target_pos = np.asarray(target_pos, dtype=np.float64).reshape(3)
    r_tgt = (
        FIXED_APPROACH_ROT_MAT
        if target_rot is None
        else np.asarray(target_rot, dtype=np.float64).reshape(3, 3)
    )

    joint_ids: list[int] = []
    qadr: list[int] = []
    dof_ids: list[int] = []
    for name in _RIGHT_ARM:
        jid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise RuntimeError(f"missing joint {name}")
        joint_ids.append(int(jid))
        qadr.append(int(env.model.jnt_qposadr[jid]))
        dof_ids.append(int(env.model.jnt_dofadr[jid]))

    for _ in range(int(max_iters)):
        mujoco.mj_forward(env.model, env.data)
        tcp = np.asarray(env.data.site_xpos[env._tcp_site_id], dtype=np.float64)
        r_cur = np.asarray(env.data.site_xmat[env._tcp_site_id], dtype=np.float64).reshape(3, 3)
        pos_e = target_pos - tcp
        ori_e = _rot_err(r_cur, r_tgt)
        if float(np.linalg.norm(pos_e)) < pos_tol and float(np.linalg.norm(ori_e)) < ori_tol:
            return env._current_control_qpos().astype(np.float32).copy()

        jacp = np.zeros((3, env.model.nv), dtype=np.float64)
        jacr = np.zeros((3, env.model.nv), dtype=np.float64)
        mujoco.mj_jacSite(env.model, env.data, jacp, jacr, int(env._tcp_site_id))
        J = np.vstack([jacp[:, dof_ids], jacr[:, dof_ids]])
        err = np.concatenate([pos_e, ori_e])
        H = J.T @ J + (float(damping) ** 2) * np.eye(7, dtype=np.float64)
        try:
            dq = np.linalg.solve(H, J.T @ err)
        except np.linalg.LinAlgError:
            return None
        for adr, dqi, jid in zip(qadr, dq, joint_ids, strict=True):
            lo, hi = env.model.jnt_range[jid]
            env.data.qpos[adr] = float(np.clip(env.data.qpos[adr] + dqi, lo, hi))
        env._apply_mimics()

    return None
