# Unoarm Reach-IK Data Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate LeRobot training episodes automatically from multiple sword-handle targets (JSON list or AABB sampling) via MuJoCo DLS IK: approach → grasp → pass execution point (right arm only).

**Architecture:** Add `gym_unoarm/ik.py` (Jacobian DLS on `right_tcp`), `data_gen/reach_ik.py` (targets → place sword → keyframes → rollout → dataset), and CLI `scripts/10_generate_reach_ik_data.py`. Reuse `data_gen/scripted.py` features/`add_frame`/interpolate. No new IK dependencies.

**Tech Stack:** Python 3.12+, MuJoCo, NumPy, Gymnasium `UnoarmEnv`, LeRobotDataset, existing `resolve_sword_pose`.

**Spec:** `docs/superpowers/specs/2026-07-27-unoarm-reach-ik-datagen-design.md`

## Global Constraints

- Right arm / sword only; shield removed (`remove_shield=True`)
- Fixed TCP orientation (approach along −Y in base frame); approach offset **6 cm** along **+Y**
- Always include execution-point stage; default pos `DEFAULT_EXECUTION_POINT_POS`
- When attached, handle ≈ TCP → execution IK targets TCP at `execution_point_pos`
- Pose jitter default **0.0** for IK path
- No new packages beyond MuJoCo already in `custom_envs/unoarm`
- Do **not** change behavior of `06_generate_scripted_data.py` / free-space scripted path
- Commits: only if the user explicitly asks (skip commit steps otherwise)

## File map

| File                                                      | Responsibility                                   |
| --------------------------------------------------------- | ------------------------------------------------ |
| `custom_envs/unoarm/gym_unoarm/ik.py`                     | DLS IK for right arm TCP pose                    |
| `custom_envs/unoarm/data_gen/reach_ik.py`                 | Targets, sword placement, keyframes, dataset gen |
| `custom_envs/unoarm/scripts/10_generate_reach_ik_data.py` | CLI                                              |
| `custom_envs/unoarm/tests/test_ik.py`                     | IK unit tests                                    |
| `custom_envs/unoarm/tests/test_reach_ik_datagen.py`       | Loader/sampler + smoke gen                       |
| `custom_envs/unoarm/data/targets_reach_ik_example.json`   | Example targets file                             |

---

### Task 1: Right-arm MuJoCo DLS IK

**Files:**

- Create: `custom_envs/unoarm/gym_unoarm/ik.py`
- Test: `custom_envs/unoarm/tests/test_ik.py`

**Interfaces:**

- Consumes: `UnoarmEnv` with `scene=reach_sword`; sites `right_tcp`; joints `Right_Joint1`…`Right_Joint7`
- Produces:
  - `FIXED_APPROACH_ROT_MAT: np.ndarray` shape `(3, 3)` — columns are TCP axes in world; +X along gripper (site), facing workspace (−Y world forward for approach)
  - `solve_right_tcp_ik(env, target_pos, *, target_rot=None, max_iters=200, pos_tol=0.008, ori_tol=0.08, damping=1e-2) -> np.ndarray | None`
    Returns raw **16-D** `CONTROL_JOINTS` qpos on success (left arm unchanged from current), else `None`. Does not change gripper values except leaving them as-is from current qpos.

**Orientation convention (lock in code + docstring):**
Build `FIXED_APPROACH_ROT_MAT` so TCP +X (gripper axis, matching `RIGHT_TCP_LOCAL`) points roughly **−Y** (into workspace), +Z roughly **+Z** (up). Use orthonormalization (`np.linalg.qr` or cross products).

- [ ] **Step 1: Write failing tests**

```python
# custom_envs/unoarm/tests/test_ik.py
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gym_unoarm.constants import SCENE_REACH_SWORD
from gym_unoarm.env import UnoarmEnv
from gym_unoarm.ik import FIXED_APPROACH_ROT_MAT, solve_right_tcp_ik


def test_fixed_approach_rot_is_orthonormal():
    R = FIXED_APPROACH_ROT_MAT
    assert R.shape == (3, 3)
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-6)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-5)


def test_ik_reaches_in_workspace_handle_default():
    env = UnoarmEnv(scene=SCENE_REACH_SWORD, remove_shield=True)
    env.reset()
    target = np.array([-0.35, -0.55, 1.0], dtype=np.float64)
    q = solve_right_tcp_ik(env, target)
    assert q is not None
    env.apply_raw_qpos(q)
    tcp = np.asarray(env.data.site_xpos[env._tcp_site_id], dtype=np.float64)
    assert np.linalg.norm(tcp - target) < 0.01
    env.close()


def test_ik_returns_none_far_away():
    env = UnoarmEnv(scene=SCENE_REACH_SWORD, remove_shield=True)
    env.reset()
    q = solve_right_tcp_ik(env, np.array([5.0, 5.0, 5.0]))
    assert q is None
    env.close()
```

- [ ] **Step 2: Run tests — expect FAIL (module missing)**

```bash
cd /home/doki/project/lerobot-main/custom_envs/unoarm
uv run pytest tests/test_ik.py -svv
```

Expected: `ModuleNotFoundError` / import error for `gym_unoarm.ik`

- [ ] **Step 3: Implement `ik.py`**

```python
# custom_envs/unoarm/gym_unoarm/ik.py
"""MuJoCo damped-least-squares IK for Unoarm right TCP."""

from __future__ import annotations

import mujoco
import numpy as np

from .constants import CONTROL_JOINTS

# Gripper axis (+X of right_tcp) points into workspace (−Y).
_x = np.array([0.0, -1.0, 0.0], dtype=np.float64)
_z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
_y = np.cross(_z, _x)
_y = _y / np.linalg.norm(_y)
_z = np.cross(_x, _y)
FIXED_APPROACH_ROT_MAT = np.column_stack([_x, _y, _z])

_RIGHT_ARM = tuple(f"Right_Joint{i}" for i in range(1, 8))


def _rot_err(r_cur: np.ndarray, r_tgt: np.ndarray) -> np.ndarray:
    # 3-vector orientation error from relative rotation.
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
    max_iters: int = 200,
    pos_tol: float = 0.008,
    ori_tol: float = 0.08,
    damping: float = 1e-2,
) -> np.ndarray | None:
    """Solve right-arm IK so ``right_tcp`` matches target pose.

    Returns clipped raw 16-D control qpos, or ``None`` on failure.
    Left arm and grippers keep their values from the env state at call time.
    """
    if env._tcp_site_id < 0:
        raise RuntimeError("env has no right_tcp site (need reach_sword scene)")
    target_pos = np.asarray(target_pos, dtype=np.float64).reshape(3)
    r_tgt = FIXED_APPROACH_ROT_MAT if target_rot is None else np.asarray(target_rot, dtype=np.float64).reshape(3, 3)

    joint_ids = []
    qadr = []
    for name in _RIGHT_ARM:
        jid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise RuntimeError(f"missing joint {name}")
        joint_ids.append(jid)
        qadr.append(int(env.model.jnt_qposadr[jid]))

    # dof indices for jac
    dof_ids = [int(env.model.jnt_dofadr[jid]) for jid in joint_ids]

    for _ in range(max_iters):
        mujoco.mj_forward(env.model, env.data)
        tcp = np.asarray(env.data.site_xpos[env._tcp_site_id], dtype=np.float64)
        r_cur = np.asarray(env.data.site_xmat[env._tcp_site_id], dtype=np.float64).reshape(3, 3)
        pos_e = target_pos - tcp
        ori_e = _rot_err(r_cur, r_tgt)
        if float(np.linalg.norm(pos_e)) < pos_tol and float(np.linalg.norm(ori_e)) < ori_tol:
            return env._current_control_qpos().astype(np.float32).copy()

        jacp = np.zeros((3, env.model.nv), dtype=np.float64)
        jacr = np.zeros((3, env.model.nv), dtype=np.float64)
        mujoco.mj_jacSite(env.model, env.data, jacp, jacr, env._tcp_site_id)
        J = np.vstack([jacp[:, dof_ids], jacr[:, dof_ids]])  # 6 x 7
        err = np.concatenate([pos_e, ori_e])
        H = J.T @ J + (damping**2) * np.eye(7)
        dq = np.linalg.solve(H, J.T @ err)
        for adr, dqi, jid in zip(qadr, dq, joint_ids, strict=True):
            lo, hi = env.model.jnt_range[jid]
            env.data.qpos[adr] = float(np.clip(env.data.qpos[adr] + dqi, lo, hi))
        env._apply_mimics()

    return None
```

Implement against real MuJoCo APIs; if `mj_jacSite` signature differs in the installed MuJoCo, adjust to the local version (check with `help(mujoco.mj_jacSite)`). Keep the public signatures above.

- [ ] **Step 4: Run tests — expect PASS**

```bash
uv run pytest tests/test_ik.py -svv
```

Expected: 3 passed

- [ ] **Step 5: Commit only if user asked**

---

### Task 2: Target load / sample + sword placement helpers

**Files:**

- Create: `custom_envs/unoarm/data_gen/reach_ik.py` (helpers first; generation in Task 3)
- Create: `custom_envs/unoarm/data/targets_reach_ik_example.json`
- Test: `custom_envs/unoarm/tests/test_reach_ik_datagen.py`

**Interfaces:**

- Produces:
  - `DEFAULT_TASK = "Reach the sword handle then pass the execution point"`
  - `load_targets_json(path: Path) -> tuple[str, list[np.ndarray], np.ndarray]`
    → `(task, list of xyz (3,), execution_point_pos (3,))`
  - `sample_targets_in_bbox(bbox_min, bbox_max, n, rng) -> list[np.ndarray]`
  - `place_sword_handle(env, handle_xyz, euler_deg=(0,0,0)) -> None`
    Uses `resolve_sword_pose`; sets `body_pos/quat` and `_sword_home_*`

- [ ] **Step 1: Write failing tests**

```python
# in test_reach_ik_datagen.py
def test_load_targets_json(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(
        '{"task": "T", "execution_point_pos": [0,0,1], "targets": [{"xyz": [-0.3,-0.5,1.0]}]}',
        encoding="utf-8",
    )
    from data_gen.reach_ik import load_targets_json
    task, targets, ep = load_targets_json(p)
    assert task == "T"
    assert len(targets) == 1
    assert np.allclose(targets[0], [-0.3, -0.5, 1.0])
    assert np.allclose(ep, [0, 0, 1])


def test_sample_targets_in_bbox():
    from data_gen.reach_ik import sample_targets_in_bbox
    rng = np.random.default_rng(0)
    pts = sample_targets_in_bbox([-0.4, -0.6, 0.9], [-0.2, -0.4, 1.1], 20, rng)
    assert len(pts) == 20
    for p in pts:
        assert (-0.4 <= p[0] <= -0.2) and (-0.6 <= p[1] <= -0.4) and (0.9 <= p[2] <= 1.1)


def test_place_sword_handle_updates_site():
    from gym_unoarm.constants import SCENE_REACH_SWORD
    from gym_unoarm.env import UnoarmEnv
    from data_gen.reach_ik import place_sword_handle
    env = UnoarmEnv(scene=SCENE_REACH_SWORD, remove_shield=True)
    env.reset()
    handle = np.array([-0.28, -0.52, 1.02], dtype=np.float64)
    place_sword_handle(env, handle)
    env.reset()  # must keep home
    xpos = np.asarray(env.data.site_xpos[env._handle_site_id], dtype=np.float64)
    assert np.allclose(xpos, handle, atol=1e-4)
    env.close()
```

- [ ] **Step 2: Run — expect FAIL**

```bash
uv run pytest tests/test_reach_ik_datagen.py -svv -k "load_targets or sample_targets or place_sword"
```

- [ ] **Step 3: Implement helpers in `reach_ik.py` + example JSON**

Example JSON:

```json
{
  "task": "Reach the sword handle then pass the execution point",
  "execution_point_pos": [-0.15, -0.35, 1.15],
  "targets": [
    { "xyz": [-0.35, -0.55, 1.0] },
    { "xyz": [-0.3, -0.5, 1.05] },
    { "xyz": [-0.25, -0.55, 1.02] }
  ]
}
```

`place_sword_handle`:

```python
def place_sword_handle(env, handle_xyz, euler_deg=(0.0, 0.0, 0.0)) -> None:
    body, quat, _hw = resolve_sword_pose(handle_pos=tuple(handle_xyz), euler_deg=tuple(euler_deg))
    body_arr = np.asarray(body, dtype=np.float64)
    quat_arr = np.asarray(quat, dtype=np.float64)
    env.model.body_pos[env._sword_body_id] = body_arr
    env.model.body_quat[env._sword_body_id] = quat_arr
    env._sword_home_pos = body_arr.copy()
    env._sword_home_quat = quat_arr.copy()
    mujoco.mj_forward(env.model, env.data)
```

- [ ] **Step 4: Run — expect PASS**

- [ ] **Step 5: Commit only if user asked**

---

### Task 3: Keyframe build + dataset generation loop

**Files:**

- Modify: `custom_envs/unoarm/data_gen/reach_ik.py`
- Test: `custom_envs/unoarm/tests/test_reach_ik_datagen.py`

**Interfaces:**

- Consumes: Task 1 IK; Task 2 helpers; `data_gen.scripted` (`dataset_features`, `add_frame`, `build_episode_actions`, `clip_raw_pose`, `images_are_black`, `RAW_ZERO`)
- Produces:
  - `@dataclass ReachIkGenConfig` with fields:
    `targets_json: Path | None`, `bbox_min/max: tuple[float,float,float] | None`, `num_targets: int`,
    `episodes_per_target: int = 1`, `segment_steps: int = 20`, `hold_steps: int = 2`,
    `pose_jitter_std: float = 0.0`, `seed: int = 0`, `output_root: Path | None`,
    `repo_id: str = "doki/unoarm_reach_ik"`, `overwrite: bool = False`,
    `execution_point_pos: tuple[float,float,float] | None = None`,
    `approach_offset_m: float = 0.06`, `ik_retries: int = 8`
  - `build_reach_ik_poses(env, handle_xyz, execution_point_pos, approach_offset_m=0.06) -> list[np.ndarray] | None`
    Keyframes (raw 16-D): home → approach (open) → grasp (open) → grasp (closed) → exec (closed) → hold uses scripted holds
  - `generate_reach_ik_dataset(cfg, *, log=print) -> Path`
    Returns output root; raises `RuntimeError` if zero episodes written

**Keyframe details:**

1. Reset/home: `RAW_ZERO` (grippers open at +raw corresponding to open — use env limits; set right gripper to **high/open** raw via denormalize of `GRIPPER_OPEN` or clip open)
2. Approach TCP = `handle + (0, +approach_offset_m, 0)`; IK; force right gripper open (normalized → raw)
3. Grasp TCP = `handle`; IK; gripper open
4. Same arm joints; right gripper **closed** (normalized `GRIPPER_CLOSE`)
5. Exec TCP = `execution_point_pos`; IK from current (attached) state; gripper closed
6. Pass list to `build_episode_actions` **without** re-wrapping `build_pose_sequence` zero bookends if home already first — either include home as first pose and skip duplicate zeros, or call `build_episode_actions` on `[approach, grasp_open, grasp_close, exec]` after starting from reset zero. Prefer: poses = `[approach_open, grasp_open, grasp_close, exec_close]` and let `build_episode_actions` prepend holds from `RAW_ZERO` as it already does.

Gripper open/close helpers:

```python
def _with_right_gripper(env, q: np.ndarray, open_: bool) -> np.ndarray:
    out = q.astype(np.float32).copy()
    # CONTROL index 15; map normalized ±1 through env._denormalize
    n = env._normalize(out)
    n[RIGHT_GRIPPER_INDEX] = GRIPPER_OPEN if open_ else GRIPPER_CLOSE
    return env._denormalize(n).astype(np.float32)
```

Generation loop sketch:

```python
for ti, handle in enumerate(targets):
    place_sword_handle(env, handle)
    env.set_execution_point(enable=True, position=execution_point_pos)
    poses = None
    for attempt in range(cfg.ik_retries):
        env.reset(options={"state": env._normalize(clip_raw_pose(env, RAW_ZERO))})
        poses = build_reach_ik_poses(env, handle, execution_point_pos, cfg.approach_offset_m)
        if poses is not None:
            break
    if poses is None:
        log(f"skip target {ti}: IK failed"); continue
    for ep in range(cfg.episodes_per_target):
        # reset, optional jitter on arm joints of poses, step actions, add_frame, save_episode
        # after last frame, optionally assert info execution_point_passed for smoke (log warn if not)
```

Env ctor:

```python
UnoarmEnv(
    scene=SCENE_REACH_SWORD,
    remove_shield=True,
    enable_execution_point=True,
    execution_point_pos=tuple(execution_point_pos),
    terminate_on_success=False,
    max_episode_steps=10_000,
    use_pointmap=True,
)
```

- [ ] **Step 1: Write integration smoke test**

```python
def test_generate_one_target_smoke(tmp_path):
    from data_gen.reach_ik import ReachIkGenConfig, generate_reach_ik_dataset
    out = tmp_path / "ds"
    # write tiny targets json with 1 point near default handle
    ...
    cfg = ReachIkGenConfig(
        targets_json=p,
        episodes_per_target=1,
        segment_steps=5,
        hold_steps=1,
        output_root=out,
        overwrite=True,
        repo_id="test/unoarm_reach_ik",
    )
    root = generate_reach_ik_dataset(cfg)
    assert root.exists()
    # meta / episode dir exists (LeRobot layout)
    assert any(out.iterdir())
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement `build_reach_ik_poses` + `generate_reach_ik_dataset`**

- [ ] **Step 4: Run smoke test — expect PASS**

```bash
uv run pytest tests/test_reach_ik_datagen.py -svv
```

- [ ] **Step 5: Commit only if user asked**

---

### Task 4: CLI `10_generate_reach_ik_data.py`

**Files:**

- Create: `custom_envs/unoarm/scripts/10_generate_reach_ik_data.py`
- Modify: `custom_envs/unoarm/README.md` — short section under data generation pointing to the new script and example JSON

**Interfaces:**

- Consumes: `ReachIkGenConfig`, `generate_reach_ik_dataset`
- CLI flags matching spec: `--targets-json` XOR (`--bbox-min` 3 floats + `--bbox-max` 3 floats + `--num-targets`); `--episodes-per-target`; `--segment-steps`; `--hold-steps`; `--pose-jitter-std` default 0; `--seed`; `--output-root`; `--repo-id`; `--overwrite`; `--execution-point` optional 3 floats; `--approach-offset`

Validation: if neither JSON nor bbox → error; if both → prefer JSON and print warning.

- [ ] **Step 1: Implement CLI** (mirror structure of `06_generate_scripted_data.py`)

- [ ] **Step 2: Dry-run help**

```bash
uv run python custom_envs/unoarm/scripts/10_generate_reach_ik_data.py --help
```

Expected: help text lists modes and fixed-orientation note (−Y approach).

- [ ] **Step 3: Generate from example (if env deps available)**

```bash
uv run python custom_envs/unoarm/scripts/10_generate_reach_ik_data.py \
  --targets-json custom_envs/unoarm/data/targets_reach_ik_example.json \
  --episodes-per-target 1 --segment-steps 10 --hold-steps 1 \
  --output-root custom_envs/unoarm/data/unoarm_reach_ik_smoke \
  --overwrite
```

Expected: log lines `saved episode…`, `dataset finalized…`, non-zero exit only on total failure.

- [ ] **Step 4: README blurb**

- [ ] **Step 5: Commit only if user asked**

---

## Spec coverage check

| Spec item                                            | Task        |
| ---------------------------------------------------- | ----------- |
| JSON targets + bbox sampling                         | 2, 4        |
| MuJoCo DLS IK + fixed orientation + 6 cm +Y approach | 1, 3        |
| Approach → grasp → execution point                   | 3           |
| Right arm only / remove shield                       | 3           |
| Reuse dataset features / add_frame                   | 3           |
| Skip failed IK; fail if zero episodes                | 3           |
| CLI entry                                            | 4           |
| Tests                                                | 1–3         |
| No change to scripted `06` path                      | (untouched) |

## Placeholder scan

None intentional. Gripper raw conversion must use `env._normalize` / `_denormalize` as shown (not hard-coded raw ≈ −0.91).

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-27-unoarm-reach-ik-datagen.md`.

**Two execution options:**

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks
2. **Inline Execution** — implement in this session task-by-task with checkpoints

Which approach?
