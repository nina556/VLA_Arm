# Unoarm Reach-IK Scripted Dataset Generation

**Date:** 2026-07-27
**Status:** Approved for planning (pending user review of this written spec)
**Scope:** Right-arm sword grasp + execution-point pass; automatic multi-target training data

## Goal

Given multiple grasp targets (explicit list and/or AABB random sampling), automatically generate LeRobot training episodes where the right arm:

1. Approaches the sword handle at that target
2. Closes the gripper and kinematically attaches
3. Moves so the **sword handle** passes the configured **execution point** (green waypoint)

No teleop / manual joint keyframes required for this path.

## Decisions (locked)

| Topic             | Choice                                                      |
| ----------------- | ----------------------------------------------------------- |
| Target input      | **Both:** JSON point list **or** AABB random sampling (`C`) |
| Trajectory stages | Always: approach → grasp → pass execution point (`B`)       |
| Arms              | Right arm / sword only (`A`)                                |
| TCP orientation   | Position IK + **fixed approach orientation** (`B`)          |
| IK method         | MuJoCo Jacobian damped least squares (no new deps)          |

## Non-goals (v1)

- Left arm / shield dual-target generation
- Web UI for clicking targets or launching jobs
- Collision-aware Cartesian planning / RRT
- Per-target full 6-DoF orientation authoring
- Real-robot bridge recording in this pipeline

## Architecture

```
targets (JSON list | bbox sample)
        │
        ▼
┌───────────────────┐
│ data_gen/reach_ik │  resolve targets → place sword → IK keyframes → joint traj
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ gym_unoarm.env    │  step / attach / execution_point sticky success / RGB+pointmap
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ dataset write     │  reuse scripted features + add_frame patterns
└───────────────────┘
```

### New pieces

| Path                                                      | Role                                                                                                        |
| --------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `custom_envs/unoarm/gym_unoarm/ik.py`                     | Right-arm DLS IK: target TCP position + fixed orientation → 7 arm joints (+ gripper open/close as scripted) |
| `custom_envs/unoarm/data_gen/reach_ik.py`                 | Target loading/sampling, sword placement, keyframe build, episode rollout, dataset export                   |
| `custom_envs/unoarm/scripts/10_generate_reach_ik_data.py` | CLI entry                                                                                                   |

### Reuse

- `UnoarmEnv` with `scene=reach_sword`, `enable_execution_point=True`, `terminate_on_success` optional (generation may run full scripted horizon instead of early terminate)
- `data_gen/scripted.py`: `dataset_features()`, `add_frame()`, interpolate/hold helpers where applicable
- Settings defaults: `DEFAULT_EXECUTION_POINT_POS`, `REACH_SUCCESS_THRESHOLD`, sword pose helpers

## Target specification

### Explicit list (`--targets-json`)

```json
{
  "task": "Reach the sword handle then pass the execution point",
  "execution_point_pos": [-0.15, -0.35, 1.15],
  "targets": [{ "xyz": [-0.35, -0.55, 1.0] }, { "xyz": [-0.25, -0.5, 1.05] }]
}
```

- `execution_point_pos` optional; default `DEFAULT_EXECUTION_POINT_POS`
- Each target is sword **handle** world XYZ (meters, robot base frame)
- Sword body orientation for placement: fixed default euler (same as current web default), unless later extended

### Random sampling (`--bbox-min` / `--bbox-max` / `--num-targets`)

- Uniform sample in axis-aligned box
- Reject / retry if IK fails (cap retries per slot)
- Seed via `--seed` for reproducibility

CLI must accept **either** list mode **or** bbox mode (not both required; if both given, prefer explicit list and warn).

## IK specification

- Site: `right_tcp`
- Controlled DoF: `Right_Joint1`…`Right_Joint7` (left arm held at start/home)
- Error: 3D position + orientation (**fixed** target rotation: gripper approach along **−Y** in the robot base frame — “reach forward toward the workspace”; exact quaternion constant defined once in `ik.py` and documented in CLI `--help`)
- Solver: damped least squares on MuJoCo analytic Jacobian; iterate until position error &lt; threshold (default **8 mm**) and orientation error below a fixed angular threshold, or max iters (e.g. 200)
- Joint limits: clamp to MJCF ranges each step
- Failure: return `None`; caller skips target

### Fixed approach orientation

Single constant rotation for all grasp IK solves (v1), as above.

**Approach keyframe:** same orientation; TCP placed **6 cm** back from the handle along **+Y** (away from the robot’s forward reach) so the arm pre-positions before the final grasp pose.

## Episode keyframe sequence

For each successful target:

1. **Home** — `START_POSE` / zero (match existing scripted convention)
2. **Approach** — IK at offset pose, gripper open
3. **Grasp** — IK at handle, gripper open → then close gripper (attach)
4. **Execution** — With sword attached, solve IK so that the **attached sword handle** lands at `execution_point_pos`: compute the TCP target as `execution_point_pos` transformed by the fixed attach offset (handle relative to TCP at grasp time), then IK that TCP pose with the same fixed orientation.
5. **Hold** — short hold after pass (default hold steps aligned with scripted gen, e.g. 2)

Interpolation: joint-space linear between keyframes (reuse `interpolate` / `hold` from scripted gen). Optional small per-episode joint jitter on arm joints only (`--pose-jitter-std`, **default 0.0** for the IK path so Cartesian accuracy stays intact; user may raise for diversity).

`episodes_per_target` (default 1): repeat the same target with optional jitter/seed offset for diversity.

## Scene sync

Before each episode:

- Place sword so **handle world position** equals the target (existing `resolve_sword_pose` / env body pos+quat APIs)
- Enable execution point; set position from config
- `remove_sword=False`; shield may be removed or left parked — prefer `remove_shield=True` for cleaner right-arm-only data unless we need it in view (default: remove shield)

## Dataset output

- Same feature schema as scripted gen: `observation.state` (16), `observation.images.top`, `observation.pointmap`, `action` (16), `task`
- Default root: `custom_envs/unoarm/data/unoarm_reach_ik_<stamp_or_name>/`
- `--overwrite` behavior aligned with scripted CLI
- Log summary: targets attempted / IK failures / episodes written

## CLI sketch

```bash
uv run python custom_envs/unoarm/scripts/10_generate_reach_ik_data.py \
  --targets-json path/to/targets.json \
  --episodes-per-target 1 \
  --overwrite

# or
uv run python custom_envs/unoarm/scripts/10_generate_reach_ik_data.py \
  --bbox-min -0.45 -0.65 0.95 \
  --bbox-max -0.20 -0.40 1.15 \
  --num-targets 40 \
  --seed 0
```

## Testing

- Unit: IK reaches a known in-workspace point within threshold; fails gracefully out of reach
- Integration: one-target smoke generates ≥1 episode; info shows attach + execution_point_passed by end when stepping the scripted actions
- Pure helpers: bbox sampling respects bounds; JSON loader validates schema

## Error handling

- Invalid JSON / empty targets → hard fail before env create
- Per-target IK failure → skip + count; continue batch
- If zero episodes written → non-zero exit
- Black-image guard: reuse scripted check if applicable

## Success criteria

1. User can supply either a point list or a bbox and get a LeRobot dataset without manual poses
2. Episodes visually/kinematically: grasp attach then handle passes execution point
3. No new heavy IK dependencies beyond MuJoCo
4. Existing keyframe scripted path (`06_…`) unchanged
