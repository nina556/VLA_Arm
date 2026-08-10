# Unoarm Table-Place Reach-IK Dataset Generation

**Date:** 2026-08-04
**Status:** Approved (user confirmed design; pending review of this written spec)
**Scope:** Web Reach-IK tab, `table_place` scene — scripted pick-and-place episodes via right-arm IK
**Supersedes (Phase 1.5 sampling):** earlier table-place design sampled peg XY; this spec locks **fixed peg + fixed circle**, randomize trajectory parameters only.

## Goal

When Web settings use scene `table_place`, the existing **Reach-IK** tab generates LeRobot training episodes that:

1. Approach and **grasp** the upright cylinder (kinematic attach)
2. **Lift** and **transport** it above the fixed green target circle **without contacting the table**
3. **Open** the gripper so the cylinder falls
4. Keep the episode **only if** it lands inside the circle (`info["success"] == True`) and lift/transport cleared the table

No teleop keyframes required. Sword Reach-IK (`reach_sword`) behavior is unchanged.

## Decisions (locked)

| Topic           | Choice                                                                                                                                                           |
| --------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Architecture    | New module `data_gen/table_place_ik.py`; Web dispatches by scene (`B`)                                                                                           |
| Peg / circle    | **Fixed** defaults (`PEG_DEFAULT_XY`, `TABLE_CIRCLE_*`)                                                                                                          |
| What is sampled | Trajectory params only: approach offset, lift/place height (± ranges)                                                                                            |
| Entry           | Web Reach-IK tab; UI adapts when `scene == table_place`                                                                                                          |
| Episode filter  | **Success only** (discard IK fail / no attach / miss circle / **lift–transport table collision**)                                                                |
| Arms            | Right arm only                                                                                                                                                   |
| TCP orientation | Reuse existing `FIXED_APPROACH_ROT_MAT` (gripper +X along −Y); same as working manual grasp                                                                      |
| IK method       | Existing `solve_right_tcp_ik` + grasp-center→TCP offset compensation                                                                                             |
| Dataset format  | Same as sword Reach-IK: `dataset_features()` / RGB + pointmap + 16-D                                                                                             |
| Table clearance | **Lift + transport only**: peg bottom (and right-arm finger meshes) must stay above the table top by ≥ `table_clearance_m`; approach/grasp may go near the table |

## Non-goals

- Randomizing peg or circle pose in v1
- CLI script (Web only for this phase; CLI can wrap the same module later)
- Left arm / dual-arm
- Collision-aware Cartesian planning
- Changing SmolVLA train entry or obs schema
- Vertical-down TCP reorientation (deferred unless current orientation proves insufficient)

## Architecture

```
Web Reach-IK (table_place)
        │
        ▼
┌─────────────────────────────┐
│ webapp/runner               │  start_reach_ik_generate → scene dispatch
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ data_gen/table_place_ik.py  │  sample params → IK keyframes → step → filter
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ gym_unoarm.env (table_place)│  attach / fall / success metrics
└─────────────┬───────────────┘
              │
              ▼
     LeRobotDataset + table_place_ik_meta.json
```

Sword path continues to call `data_gen/reach_ik.generate_reach_ik_dataset`.

## Trajectory keyframes

Per attempt, sample continuous params (defaults tunable in config / Web):

| Param               | Role                                                                          | Suggested default range      |
| ------------------- | ----------------------------------------------------------------------------- | ---------------------------- |
| `approach_offset_y` | Pre-grasp backup along +Y from grasp TCP target                               | `[0.04, 0.10]` m             |
| `lift_z`            | Desired **peg/grasp-center** world Z after lift (converted to TCP via offset) | table-top + `[0.12, 0.20]` m |
| `place_z`           | Desired **peg/grasp-center** world Z above circle before release              | table-top + `[0.10, 0.18]` m |

Keyframe sequence (raw 16-D poses, gripper open/close as noted):

1. **approach** — open; TCP at grasp target + `(0, approach_offset_y, 0)`
2. **pregrasp** — open; TCP such that finger midpoint aligns with peg center
3. **grasp_close** — same arm config, gripper closed (attach)
4. **lift** — closed; raise to `lift_z` (peg follows while attached); **must clear table**
5. **transport** — closed; XY over circle center, Z = `place_z`; **must clear table**
6. **release** — open at place pose
7. **settle** — hold open for `settle_steps` (default ≥ fall budget) until peg not falling

Interpolate with existing `build_episode_actions(segment_steps, hold_steps)`.

### Table clearance (lift + transport)

Approach and grasp may bring the jaw near the table (peg rests on the top).
**After grasp, through lift and transport until release**, every interpolated step must satisfy:

1. **Peg bottom above table:**
   `peg_center_z - PEG_HALF_HEIGHT >= table_place_top_z() + table_clearance_m`
2. **No finger–table penetration:** signed geom distance from each active right-finger mesh to the table geom is `>= -eps` (no deep penetration); optionally require `>= table_clearance_m` for the lowest finger point if cheap to compute.

Default `table_clearance_m = 0.01` (1 cm). Config / Web expose this value.

Enforcement:

- Prefer **planning**: sample `lift_z` / `place_z` so the peg-bottom clearance is satisfied by construction (`lift_z >= table_top + PEG_HALF_HEIGHT + table_clearance_m`, same for `place_z`).
- **Runtime check** on every step from first lift frame through last pre-release frame; if violated → discard attempt (do not save).

Approach/pregrasp/grasp_close steps are **not** subject to this clearance rule.

### Grasp-center ↔ TCP compensation

`solve_right_tcp_ik` targets `right_tcp`, but attach keys off finger-mesh midpoint. Before solving grasp:

1. Measure `offset = grasp_center_world - right_tcp_world` at a seeded pose (open gripper near peg).
2. Solve IK for `target_tcp = peg_center - offset` (and similarly for approach / lift / place TCP targets derived from desired grasp-center / peg paths).

If attach does not occur after `grasp_close` hold, treat attempt as failed.

## Success / discard rules

Keep episode only if **all** hold after settle:

1. At least one step had `peg_attached == True` after `grasp_close`
2. Final `info["success"]` is True (`not peg_attached` and planar distance to circle center `< TABLE_CIRCLE_RADIUS`)
3. Peg is not still falling (`peg_falling == False`)
4. Every lift/transport step passed the **table clearance** checks above

Otherwise skip (do not `save_episode`). Resample params until `num_episodes` successes or `max_attempts` (e.g. `max(num_episodes * 25, num_episodes + 50)`).

## Config & meta

`TablePlaceIkGenConfig` (fields):

- `num_episodes`, `episodes_per_param_set` (default 1), `segment_steps`, `hold_steps`, `settle_steps`
- `approach_offset_y_range`, `lift_z_range`, `place_z_range`
- `table_clearance_m` (default `0.01`)
- `pose_jitter_std`, `seed`, `ik_retries`
- `output_root`, `repo_id`, `overwrite`, `task`
- Fixed peg/circle come from env constants (not config overrides in v1)

Meta file `table_place_ik_meta.json`:

- `mode`: `"table_place_pick_place"`
- `task`, `peg_xy`, `circle_center`, `circle_radius`
- `num_episodes_written`, `num_attempts_skipped`
- per-episode: sampled params, frame count, final `place_distance` / `success`

## Web UI

When settings `scene == "table_place"`:

- Hide sword JSON / bbox target blocks
- Show table-place controls: `num_episodes`, output name, seed, segment/hold/settle steps, approach/lift/place ranges, `table_clearance_m`
- Status copy: pick-place generation (replace the Phase-1 placeholder warning)
- Generate POST still `/api/reach-ik/generate`; runner branches on scene

Replay: reuse Reach-IK replay against written datasets; list/info should surface `table_place_ik_meta.json` when present. 3D scene already supports peg + circle payloads.

## Testing

`custom_envs/unoarm/tests/test_table_place_ik_datagen.py`:

- Keyframe builder returns poses or `None` on IK fail
- One successful episode writes frames + meta; failed attempt does not increment dataset episodes
- Success filter rejects miss-circle / no-attach / **lift–transport table clearance** cases (can force via stubbed params or monkeypatch if needed)
- Runner dispatch: `table_place` calls table-place generator (unit or light integration)

## Out of scope follow-ups

- CLI wrapper script
- Sample peg XY on table (revert toward original Phase-1 sampling note)
- Vertical TCP orientation
- Web live preview of IK solves before batch write
