# Reach-IK grasp-only + sequential replay Implementation Plan

> **For agentic workers:** Implement task-by-task. Steps use checkbox syntax.

**Goal:** Grasp-only Reach-IK datagen (no execution point), sequential multi-episode replay with sword+red marker at each episode handle, Three.js pose sync.

**Architecture:** Shorten IK keyframes in `data_gen/reach_ik.py`; extend `_reach_ik_replay_loop` to iterate episodes; pin sword pose in `scene.js` when WS scene position changes.

**Tech Stack:** Python (MuJoCo/LeRobot), FastAPI runner, Three.js web UI

## Global Constraints

- No execution-point segment in generated trajectories this round
- Replay starts at selected episode index and auto-advances to end; Stop aborts
- Dataset features remain `dataset_features()` (SmolVLA-compatible)

---

### Task 1: Grasp-only generation

**Files:** `custom_envs/unoarm/data_gen/reach_ik.py`, `tests/test_reach_ik_datagen.py`, CLI/Web consumers

### Task 2: Sequential replay + sword place + exec off

**Files:** `custom_envs/unoarm/webapp/runner.py`

### Task 3: Three.js sword pin on pose change + Web UI cleanup

**Files:** `scene.js`, `reach_ik.js`, `index.html`, `main.js` cache bump
