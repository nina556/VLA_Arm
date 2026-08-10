# Table-Place Reach-IK Datagen Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When Web settings use `table_place`, the Reach-IK tab generates success-only pick-and-place LeRobot episodes (grasp fixed peg → lift/transport without hitting the table → release into fixed green circle).

**Architecture:** New `data_gen/table_place_ik.py` owns IK keyframes, clearance checks, and dataset write/filter. `webapp/runner.py` dispatches generate by scene. Reach-IK HTML/JS swaps sword target UI for table-place params when `scene === "table_place"`. Sword `reach_ik.py` stays unchanged.

**Tech Stack:** MuJoCo DLS IK (`solve_right_tcp_ik`), `UnoarmEnv` table_place attach/fall, LeRobotDataset, FastAPI runner, static Reach-IK tab.

**Spec:** `docs/superpowers/specs/2026-08-04-unoarm-table-place-ik-datagen-design.md`

## Global Constraints

- Scene id: `table_place`; peg XY and circle from `PEG_DEFAULT_XY` / `TABLE_CIRCLE_*` (fixed in v1)
- Sample trajectory params only: `approach_offset_y`, `lift_z`, `place_z`
- Success-only writes; discard IK fail / no attach / miss circle / lift–transport table clearance fail
- Table clearance applies **only** from first lift frame through last pre-release frame
- Default `table_clearance_m = 0.01`
- TCP orientation: existing `FIXED_APPROACH_ROT_MAT`
- Task string default: `TASK_TABLE_PLACE` (`Pick up the cylinder and place it on the target circle`)
- Do not commit unless the user explicitly asks
- Prefer `uv run` / project venv Python under `custom_envs/unoarm` tests (same pattern as existing unoarm tests)

## File map

| File | Responsibility |
|------|----------------|
| `custom_envs/unoarm/data_gen/table_place_ik.py` | Config, param sampling, TCP offset, clearance, keyframes, generate, meta |
| `custom_envs/unoarm/tests/test_table_place_ik_datagen.py` | Unit + smoke tests for the module |
| `custom_envs/unoarm/data_gen/reach_ik.py` | Extend `list_reach_ik_datasets` / info helpers to recognize `table_place_ik_meta.json` |
| `custom_envs/unoarm/webapp/runner.py` | Scene dispatch in `start_reach_ik_generate` / generate loop |
| `custom_envs/unoarm/static/web/index.html` | Table-place Reach-IK form block |
| `custom_envs/unoarm/static/web/js/reach_ik.js` | Scene-aware UI + generate payload |

---

### Task 1: Clearance + TCP-offset helpers

**Files:**
- Create: `custom_envs/unoarm/data_gen/table_place_ik.py`
- Test: `custom_envs/unoarm/tests/test_table_place_ik_datagen.py`

**Interfaces:**
- Produces:
  - `TABLE_PLACE_IK_META_NAME = "table_place_ik_meta.json"`
  - `DEFAULT_TABLE_CLEARANCE_M = 0.01`
  - `TablePlaceIkParams` dataclass: `approach_offset_y: float`, `lift_z: float`, `place_z: float`
  - `sample_table_place_ik_params(rng, *, approach_offset_y_range, lift_z_range, place_z_range, table_top_z, peg_half_height, table_clearance_m) -> TablePlaceIkParams`
  - `grasp_center_tcp_offset(env) -> np.ndarray` shape `(3,)`
  - `peg_bottom_clears_table(env, table_clearance_m) -> bool`
  - `lift_transport_clears_table(env, table_clearance_m) -> bool` (peg bottom + no deep finger/table penetration)

- [ ] **Step 1: Write failing tests**

```python
# tests/test_table_place_ik_datagen.py
from data_gen.table_place_ik import (
    sample_table_place_ik_params,
    grasp_center_tcp_offset,
    peg_bottom_clears_table,
    lift_transport_clears_table,
)
from gym_unoarm.constants import (
    PEG_HALF_HEIGHT,
    SCENE_TABLE_PLACE,
    table_place_top_z,
)
from gym_unoarm.env import UnoarmEnv

def test_sample_params_respect_clearance_floor():
    rng = np.random.default_rng(0)
    top = table_place_top_z()
    clearance = 0.01
    floor = top + PEG_HALF_HEIGHT + clearance
    for _ in range(40):
        p = sample_table_place_ik_params(
            rng,
            approach_offset_y_range=(0.04, 0.10),
            lift_z_range=(floor - 0.05, floor + 0.08),  # may include illegal lows
            place_z_range=(floor - 0.05, floor + 0.08),
            table_top_z=top,
            peg_half_height=PEG_HALF_HEIGHT,
            table_clearance_m=clearance,
        )
        assert p.lift_z >= floor - 1e-9
        assert p.place_z >= floor - 1e-9
        assert 0.04 <= p.approach_offset_y <= 0.10

def test_peg_bottom_clearance_and_tcp_offset():
    env = UnoarmEnv(scene=SCENE_TABLE_PLACE, use_pointmap=False)
    env.reset()
    off = grasp_center_tcp_offset(env)
    assert off.shape == (3,)
    # Peg resting on table → bottom ~ table top → fails clearance
    assert peg_bottom_clears_table(env, 0.01) is False
    env.model.body_pos[env._peg_body_id, 2] = table_place_top_z() + PEG_HALF_HEIGHT + 0.05
    import mujoco
    mujoco.mj_forward(env.model, env.data)
    assert peg_bottom_clears_table(env, 0.01) is True
    env.close()
```

- [ ] **Step 2: Run tests — expect import/fail**

Run: `cd custom_envs/unoarm && /home/doki/project/lerobot-main/.venv/bin/python -m pytest tests/test_table_place_ik_datagen.py::test_sample_params_respect_clearance_floor tests/test_table_place_ik_datagen.py::test_peg_bottom_clearance_and_tcp_offset -svv`

Expected: FAIL (module missing)

- [ ] **Step 3: Implement helpers in `table_place_ik.py`**

```python
@dataclass
class TablePlaceIkParams:
    approach_offset_y: float
    lift_z: float
    place_z: float

def sample_table_place_ik_params(...):
    floor = float(table_top_z) + float(peg_half_height) + float(table_clearance_m)
    ay = float(rng.uniform(*approach_offset_y_range))
    lo_l, hi_l = lift_z_range
    lo_p, hi_p = place_z_range
    lift_z = float(rng.uniform(max(lo_l, floor), max(hi_l, floor)))
    place_z = float(rng.uniform(max(lo_p, floor), max(hi_p, floor)))
    return TablePlaceIkParams(ay, lift_z, place_z)

def grasp_center_tcp_offset(env) -> np.ndarray:
    gc = env._grasp_center_world()
    tcp = np.asarray(env.data.site_xpos[env._tcp_site_id], dtype=np.float64)
    if gc is None:
        return np.zeros(3, dtype=np.float64)
    return np.asarray(gc, dtype=np.float64) - tcp

def peg_bottom_clears_table(env, table_clearance_m: float) -> bool:
    peg = env._peg_center_world()
    if peg is None:
        return False
    bottom = float(peg[2]) - float(PEG_HALF_HEIGHT)
    return bottom >= table_place_top_z() + float(table_clearance_m) - 1e-9

def lift_transport_clears_table(env, table_clearance_m: float) -> bool:
    if not peg_bottom_clears_table(env, table_clearance_m):
        return False
    # Optional: mj_geomDistance each right finger geom vs table_geom; reject if < -1e-3
    ...
```

Clamp sampled Z ranges so empty ranges cannot occur: if `hi < floor`, use `floor`.

- [ ] **Step 4: Re-run tests — expect PASS**

Run: same pytest command as Step 2  
Expected: PASS

---

### Task 2: Keyframe IK builder

**Files:**
- Modify: `custom_envs/unoarm/data_gen/table_place_ik.py`
- Test: `custom_envs/unoarm/tests/test_table_place_ik_datagen.py`

**Interfaces:**
- Consumes: Task 1 helpers; `solve_right_tcp_ik`; `GRIPPER_OPEN` / `GRIPPER_CLOSE`; `clip_raw_pose`
- Produces:
  - `build_table_place_ik_poses(env, params: TablePlaceIkParams) -> list[np.ndarray] | None`
  - Returns 6 raw 16-D poses: `[approach, pregrasp_open, grasp_close, lift, transport, release_open]`  
    (settle is hold of release, not a separate IK solve)

- [ ] **Step 1: Write failing test**

```python
def test_build_table_place_ik_poses_shape():
    env = UnoarmEnv(scene=SCENE_TABLE_PLACE, use_pointmap=False)
    env.reset()
    top = table_place_top_z()
    floor = top + PEG_HALF_HEIGHT + 0.01
    params = TablePlaceIkParams(
        approach_offset_y=0.06,
        lift_z=floor + 0.04,
        place_z=floor + 0.03,
    )
    poses = build_table_place_ik_poses(env, params)
    assert poses is not None
    assert len(poses) == 6
    for p in poses:
        assert p.shape == (16,)
    # Gripper: open, open, close, close, close, open
    assert poses[0][RIGHT_GRIPPER_INDEX] > poses[2][RIGHT_GRIPPER_INDEX]
    assert poses[5][RIGHT_GRIPPER_INDEX] > poses[2][RIGHT_GRIPPER_INDEX]
    env.close()
```

- [ ] **Step 2: Run test — expect FAIL** (`build_table_place_ik_poses` missing)

- [ ] **Step 3: Implement `build_table_place_ik_poses`**

Algorithm (mirror sword: solve grasp first, then approach from seed):

1. `env.reset()` already placed peg at default XY; do not move peg.  
2. Measure `offset = grasp_center_tcp_offset(env)` at reset (or after a mid-reach seed if offset is unreliable at home — if `||offset||` tiny, apply a temporary open-gripper IK near peg once and remeasure).  
3. `peg = peg_center`; `circle_xy = TABLE_CIRCLE_CENTER_XY`.  
4. `grasp_tcp = peg - offset`  
5. `approach_tcp = grasp_tcp + [0, params.approach_offset_y, 0]`  
6. `lift_tcp = [peg[0], peg[1], params.lift_z] - offset` (XY stay at peg until transport)  
7. `place_tcp = [circle_xy[0], circle_xy[1], params.place_z] - offset`  
8. Solve IK sequence with open/close gripper helpers (same pattern as `reach_ik._with_right_gripper`).  
9. Any `None` IK → return `None`.

- [ ] **Step 4: Run test — expect PASS**

Run: `... pytest tests/test_table_place_ik_datagen.py::test_build_table_place_ik_poses_shape -svv`

---

### Task 3: Dataset generator (success + clearance filter)

**Files:**
- Modify: `custom_envs/unoarm/data_gen/table_place_ik.py`
- Test: `custom_envs/unoarm/tests/test_table_place_ik_datagen.py`

**Interfaces:**
- Produces:
  - `@dataclass TablePlaceIkGenConfig` with fields from the spec (`num_episodes`, ranges, `table_clearance_m`, `settle_steps` default `PEG_FALL_MAX_STEPS`, `segment_steps`, `hold_steps`, `seed`, `output_root`, `repo_id`, `overwrite`, `task`, `ik_retries`, `pose_jitter_std`, `max_attempts` optional)
  - `generate_table_place_ik_dataset(config, *, log=print) -> Path`
  - Writes `table_place_ik_meta.json`

- [ ] **Step 1: Write failing smoke test**

```python
def test_generate_one_successful_episode(tmp_path: Path):
    top = table_place_top_z()
    floor = top + PEG_HALF_HEIGHT + 0.01
    cfg = TablePlaceIkGenConfig(
        num_episodes=1,
        segment_steps=8,
        hold_steps=1,
        settle_steps=40,
        approach_offset_y_range=(0.05, 0.07),
        lift_z_range=(floor + 0.03, floor + 0.05),
        place_z_range=(floor + 0.02, floor + 0.04),
        table_clearance_m=0.01,
        seed=0,
        output_root=tmp_path / "ds",
        repo_id="local/table_place_ik_test",
        overwrite=True,
        pose_jitter_std=0.0,
    )
    out = generate_table_place_ik_dataset(cfg)
    assert out.is_dir()
    meta = json.loads((out / "table_place_ik_meta.json").read_text())
    assert meta["mode"] == "table_place_pick_place"
    assert meta["num_episodes_written"] == 1
    assert len(meta["episodes"]) == 1
    assert meta["episodes"][0]["success"] is True
```

- [ ] **Step 2: Run test — expect FAIL**

- [ ] **Step 3: Implement `generate_table_place_ik_dataset`**

Loop outline:

```text
create LeRobotDataset + UnoarmEnv(scene=table_place, use_pointmap=True, ...)
while written < num_episodes and attempts < max_attempts:
  sample params
  reset; build poses (retry ik_retries)
  if poses is None: skip; continue
  build_episode_actions(poses, segment_steps, hold_steps, start=home)
  append settle: repeat last open action settle_steps times
  saw_attach = False
  clearance_ok = True
  in_lift_transport = track by action index ranges from keyframe boundaries
  for each action:
    step; record frame into buffers (do NOT save_episode yet)
    if peg_attached: saw_attach = True
    if in_lift_transport and not lift_transport_clears_table(...): clearance_ok = False
  after loop: if saw_attach and clearance_ok and info.success and not peg_falling:
    flush buffered frames via add_frame; save_episode; append meta
  else: discard buffers; skipped += 1
finalize; require written > 0 else raise RuntimeError
```

Keyframe index mapping for clearance: after `grasp_close` hold ends, until `release` begins.

Reuse `dataset_features`, `add_frame`, `build_episode_actions`, `clip_raw_pose`, `images_are_black` from `data_gen.scripted`.

- [ ] **Step 4: Run smoke test — expect PASS**

Run: `... pytest tests/test_table_place_ik_datagen.py::test_generate_one_successful_episode -svv`  
Expected: PASS (may take tens of seconds due to rendering)

- [ ] **Step 5: Add discard test**

Force clearance failure by monkeypatching `lift_transport_clears_table` to return `False` always; assert `num_episodes_written == 0` raises or with `num_episodes=1, max_attempts=3` raises `RuntimeError` and no episode dir growth. Prefer asserting generator raises when zero written (same as sword Reach-IK).

---

### Task 4: Dataset list / meta discovery

**Files:**
- Modify: `custom_envs/unoarm/data_gen/reach_ik.py` (`list_reach_ik_datasets`, optionally `load_reach_ik_meta` dual-read)
- Modify: `custom_envs/unoarm/webapp/runner.py` `get_reach_ik_dataset_info` if needed
- Test: extend `tests/test_table_place_ik_datagen.py` or `tests/test_reach_ik_datagen.py`

**Interfaces:**
- Produces: listing entries with `kind: "table_place_ik"` when `table_place_ik_meta.json` exists
- `load_table_place_ik_meta(root) -> dict | None` can live in `table_place_ik.py`; list function imports it

- [ ] **Step 1: Write test** that creates a fake dir with only `table_place_ik_meta.json` + minimal `meta/info.json` and asserts `list_reach_ik_datasets` returns `kind == "table_place_ik"`

- [ ] **Step 2: Implement listing branch**

In `list_reach_ik_datasets`, also check `child / "table_place_ik_meta.json"`; set `kind = "table_place_ik"`; read episode count from meta `episodes` list or `num_episodes_written`.

- [ ] **Step 3: Run related tests — PASS**

---

### Task 5: Web runner dispatch

**Files:**
- Modify: `custom_envs/unoarm/webapp/runner.py` (`start_reach_ik_generate`, `_reach_ik_generate_loop`)

**Interfaces:**
- Consumes: `TablePlaceIkGenConfig`, `generate_table_place_ik_dataset`
- When `self.cfg.scene == SCENE_TABLE_PLACE` (or `settings scene`), build table-place config from params and run that generator; else existing sword path

- [ ] **Step 1: Branch `start_reach_ik_generate`**

```python
from gym_unoarm.constants import SCENE_TABLE_PLACE, TASK_TABLE_PLACE
from data_gen.table_place_ik import TablePlaceIkGenConfig, generate_table_place_ik_dataset

scene = str(getattr(self.cfg, "scene", "") or "")
if scene == SCENE_TABLE_PLACE:
    # parse num_episodes, ranges, table_clearance_m, settle_steps, ...
    cfg = TablePlaceIkGenConfig(...)
    thread = threading.Thread(target=self._table_place_ik_generate_loop, args=(cfg,), daemon=True)
    ...
    return {"ok": True, "state": "running", "output_root": str(output_root), "kind": "table_place_ik"}
# else existing json|bbox sword path
```

Add `_table_place_ik_generate_loop` mirroring `_reach_ik_generate_loop` but calling `generate_table_place_ik_dataset`.

Param keys (Web → config):

| Web field | Config |
|-----------|--------|
| `num_episodes` | `num_episodes` |
| `approach_offset_y_min/max` | `approach_offset_y_range` |
| `lift_z_min/max` | `lift_z_range` (absolute world Z) |
| `place_z_min/max` | `place_z_range` |
| `table_clearance_m` | `table_clearance_m` |
| `settle_steps` | `settle_steps` |
| `segment_steps`, `hold_steps`, `seed`, `output_name`, `overwrite`, `pose_jitter_std`, `task` | same |

Default absolute Z ranges on the server if omitted: compute from `table_place_top_z()` + half height + clearance + margins matching the spec table.

- [ ] **Step 2: Manual sanity** — call `start_reach_ik_generate` logic via a small unit test that mocks scene and asserts `TablePlaceIkGenConfig` construction (optional). Prefer a focused test of a pure helper `table_place_ik_config_from_params(params) -> TablePlaceIkGenConfig` extracted in `table_place_ik.py` or `runner.py`.

---

### Task 6: Web UI for table_place

**Files:**
- Modify: `custom_envs/unoarm/static/web/index.html`
- Modify: `custom_envs/unoarm/static/web/js/reach_ik.js`

- [ ] **Step 1: HTML**

Inside `#reachIkPanel`:

- Wrap existing sword JSON/AABB blocks in `#rikSwordBlocks`
- Add `#rikTablePlaceBlock` (hidden by default) with inputs:
  - `rik_tp_num_episodes` (default 20)
  - `rik_tp_approach_min` / `rik_tp_approach_max` (0.04 / 0.10)
  - `rik_tp_lift_min` / `rik_tp_lift_max` (document as world Z; prefill via JS from `/api/settings` or fixed defaults ~0.95–1.05 once known from `table_place_top_z≈0.82`)
  - `rik_tp_place_min` / `rik_tp_place_max`
  - `rik_tp_clearance` (0.01)
  - `rik_tp_settle` (60)
- Update sword hint vs table-place hint paragraphs (`#rikHintSword` / `#rikHintTable`)

Shared controls (segment/hold/output/overwrite) stay visible for both.

- [ ] **Step 2: JS scene sync**

Replace placeholder `warnIfTablePlace` with `syncReachIkSceneUi(scene)`:

- `table_place`: show `#rikTablePlaceBlock`, hide `#rikSwordBlocks`, update hint to pick-place copy  
- else: opposite  

Call on setup and when settings are saved (hook existing settings refresh if available; otherwise re-fetch `/api/settings` when Reach-IK mode opens).

- [ ] **Step 3: Generate payload**

```javascript
if (scene === "table_place") {
  payload = {
    mode: "table_place",
    num_episodes: num("rik_tp_num_episodes", 20),
    approach_offset_y_min: num("rik_tp_approach_min", 0.04),
    approach_offset_y_max: num("rik_tp_approach_max", 0.10),
    lift_z_min: num("rik_tp_lift_min", ...),
    lift_z_max: num("rik_tp_lift_max", ...),
    place_z_min: num("rik_tp_place_min", ...),
    place_z_max: num("rik_tp_place_max", ...),
    table_clearance_m: num("rik_tp_clearance", 0.01),
    settle_steps: num("rik_tp_settle", 60),
    segment_steps: num("rik_segment", 20),
    hold_steps: num("rik_hold", 2),
    seed: num("rik_seed", 0),  // add seed to shared row if only in bbox today
    output_name: ...,
    overwrite: ...,
    task: "Pick up the cylinder and place it on the target circle",
  };
}
```

Ensure seed is available in shared UI for table-place (move seed out of bbox-only row or duplicate).

- [ ] **Step 4: Runner accepts `mode: "table_place"`** without requiring json/bbox (Task 5).

- [ ] **Step 5: Bump cache query** on `reach_ik.js` import in `main.js` if present (`?v=tableplace2`).

---

## Spec coverage checklist

| Spec requirement | Task |
|------------------|------|
| New `table_place_ik.py` module | 1–3 |
| Fixed peg/circle; sample approach/lift/place | 1, 3 |
| Keyframes approach→…→release + settle | 2–3 |
| Grasp-center TCP offset | 1–2 |
| Success-only + attach + settle | 3 |
| Lift/transport table clearance | 1, 3 |
| Meta `table_place_ik_meta.json` | 3–4 |
| Web scene-linked UI | 6 |
| Runner dispatch | 5 |
| Tests | 1–4 |

## Self-review notes

- No CLI task (spec non-goal).  
- Replay 3D already has peg/circle; listing kind update is enough for v1.  
- Commit steps omitted per global constraint (user must request commits).

---

Plan complete and saved to `docs/superpowers/plans/2026-08-04-unoarm-table-place-ik-datagen.md`.

**Two execution options:**

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks  
2. **Inline Execution** — implement in this session with executing-plans checkpoints  

Which approach?
