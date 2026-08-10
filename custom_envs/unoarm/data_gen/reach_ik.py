"""Reach-IK scripted dataset generation: multi-target grasp (single fixed point).

Given sword-handle XYZ targets (JSON list or AABB sampling), solves right-arm
MuJoCo DLS IK keyframes (approach → grasp) and writes a LeRobot dataset.
Reuses ``data_gen.scripted`` features / frame helpers. Execution-point motion
is intentionally out of scope for this grasp-only mode.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from gym_unoarm.constants import (
    FPS,
    GRIPPER_CLOSE,
    GRIPPER_OPEN,
    RIGHT_GRIPPER_INDEX,
    SCENE_REACH_SWORD,
)
from gym_unoarm.ik import solve_right_tcp_ik
from gym_unoarm.sword_pose import resolve_sword_pose

from .scripted import (
    add_frame,
    build_episode_actions,
    clip_raw_pose,
    dataset_features,
    images_are_black,
)
from .table_place_ik import TABLE_PLACE_IK_META_NAME, load_table_place_ik_meta

LogFn = Callable[[str], None]

DEFAULT_TASK = "Grasp the sword handle"
DEFAULT_SEGMENT_STEPS = 20
DEFAULT_HOLD_STEPS = 2
DEFAULT_APPROACH_OFFSET_M = 0.06
DEFAULT_IK_RETRIES = 8
REACH_IK_META_NAME = "reach_ik_meta.json"
DATA_ROOT = Path(__file__).resolve().parents[1] / "data"


@dataclass
class ReachIkGenConfig:
    targets_json: Path | None = None
    bbox_min: tuple[float, float, float] | None = None
    bbox_max: tuple[float, float, float] | None = None
    num_targets: int = 0
    episodes_per_target: int = 1
    segment_steps: int = DEFAULT_SEGMENT_STEPS
    hold_steps: int = DEFAULT_HOLD_STEPS
    pose_jitter_std: float = 0.0
    seed: int = 0
    output_root: Path | None = None
    repo_id: str = "doki/unoarm_reach_ik"
    overwrite: bool = False
    # Kept for API/CLI compatibility; ignored in grasp-only generation.
    execution_point_pos: tuple[float, float, float] | None = None
    approach_offset_m: float = DEFAULT_APPROACH_OFFSET_M
    ik_retries: int = DEFAULT_IK_RETRIES
    task: str = DEFAULT_TASK


def load_targets_json(path: Path) -> tuple[str, list[np.ndarray]]:
    """Load task and handle XYZ list from JSON. Extra keys (e.g. execution_point) ignored."""
    if not path.exists():
        raise FileNotFoundError(f"Targets JSON not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "targets" not in data:
        raise ValueError("JSON must be an object with a 'targets' list")
    task = str(data.get("task") or DEFAULT_TASK)
    raw_targets = data["targets"]
    if not isinstance(raw_targets, list) or len(raw_targets) == 0:
        raise ValueError("'targets' must be a non-empty list")
    targets: list[np.ndarray] = []
    for i, item in enumerate(raw_targets):
        xyz = item.get("xyz") if isinstance(item, dict) else item
        try:
            vec = np.asarray([float(x) for x in xyz], dtype=np.float64).reshape(3)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"target #{i} must be xyz of length 3") from exc
        if vec.shape != (3,):
            raise ValueError(f"target #{i} must be xyz of length 3")
        targets.append(vec)
    return task, targets


def sample_targets_in_bbox(
    bbox_min: tuple[float, float, float] | list[float] | np.ndarray,
    bbox_max: tuple[float, float, float] | list[float] | np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    if n < 1:
        raise ValueError(f"num_targets must be >= 1, got {n}")
    lo = np.asarray(bbox_min, dtype=np.float64).reshape(3)
    hi = np.asarray(bbox_max, dtype=np.float64).reshape(3)
    if np.any(hi < lo):
        raise ValueError(f"bbox_max must be >= bbox_min elementwise: {lo} vs {hi}")
    pts = rng.uniform(lo, hi, size=(n, 3)).astype(np.float64)
    return [pts[i].copy() for i in range(n)]


def place_sword_handle(
    env,
    handle_xyz: np.ndarray | tuple[float, float, float] | list[float],
    euler_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    """Place sword so handle world XYZ matches ``handle_xyz``; update home pose."""
    if int(getattr(env, "_sword_body_id", -1)) < 0:
        raise RuntimeError("env has no sword body (need reach_sword scene)")
    handle = tuple(float(x) for x in np.asarray(handle_xyz, dtype=np.float64).reshape(3))
    body, quat, _hw = resolve_sword_pose(handle_pos=handle, euler_deg=euler_deg)
    body_arr = np.asarray(body, dtype=np.float64)
    quat_arr = np.asarray(quat, dtype=np.float64)
    env.model.body_pos[env._sword_body_id] = body_arr
    env.model.body_quat[env._sword_body_id] = quat_arr
    env._sword_home_pos = body_arr.copy()
    env._sword_home_quat = quat_arr.copy()
    mujoco.mj_forward(env.model, env.data)


def _with_right_gripper(env, q: np.ndarray, *, open_: bool) -> np.ndarray:
    out = np.asarray(q, dtype=np.float32).reshape(16).copy()
    n = env._normalize(out)
    n[RIGHT_GRIPPER_INDEX] = GRIPPER_OPEN if open_ else GRIPPER_CLOSE
    return env._denormalize(n).astype(np.float32)


def build_reach_ik_poses(
    env,
    handle_xyz: np.ndarray,
    approach_offset_m: float = DEFAULT_APPROACH_OFFSET_M,
) -> list[np.ndarray] | None:
    """Solve approach / grasp keyframes (raw 16-D). None on IK failure.

    Assumes env is at a mid-range start (typical ``env.reset()``) and sword is
    already placed at ``handle_xyz``. Grasp is solved first (better seed), then
    approach is solved by backing up along +Y from the grasp configuration.
    """
    handle = np.asarray(handle_xyz, dtype=np.float64).reshape(3)
    approach_pos = handle + np.array([0.0, float(approach_offset_m), 0.0], dtype=np.float64)

    # Grasp first — cold start to the offset pose often fails; backing up from
    # a successful grasp seed is reliable.
    q_grasp = solve_right_tcp_ik(env, handle)
    if q_grasp is None:
        return None
    env.apply_raw_qpos(_with_right_gripper(env, q_grasp, open_=True))

    q_approach = solve_right_tcp_ik(env, approach_pos)
    if q_approach is None:
        # Fall back: small offset or reuse grasp as approach.
        q_approach = q_grasp.copy()

    # Re-seed grasp from approach so the path is continuous.
    env.apply_raw_qpos(_with_right_gripper(env, q_approach, open_=True))
    q_grasp2 = solve_right_tcp_ik(env, handle)
    if q_grasp2 is None:
        q_grasp2 = q_grasp
    grasp_open = _with_right_gripper(env, q_grasp2, open_=True)
    grasp_close = _with_right_gripper(env, q_grasp2, open_=False)

    return [
        clip_raw_pose(env, _with_right_gripper(env, q_approach, open_=True)),
        clip_raw_pose(env, grasp_open),
        clip_raw_pose(env, grasp_close),
    ]


def _resolve_targets(cfg: ReachIkGenConfig, rng: np.random.Generator) -> tuple[str, list[np.ndarray] | None]:
    """Return (task, targets).

    For AABB mode ``targets`` is ``None`` — the generator resamples until
    ``num_targets`` successful IK solutions (failed draws are skipped/retried).
    """
    if cfg.targets_json is not None and (cfg.bbox_min is not None or cfg.bbox_max is not None):
        # Prefer explicit list when both provided (caller should warn).
        task, targets = load_targets_json(Path(cfg.targets_json))
    elif cfg.targets_json is not None:
        task, targets = load_targets_json(Path(cfg.targets_json))
    elif cfg.bbox_min is not None and cfg.bbox_max is not None:
        if cfg.num_targets < 1:
            raise ValueError("bbox mode requires num_targets >= 1")
        task = cfg.task or DEFAULT_TASK
        targets = None
    else:
        raise ValueError("Provide targets_json or both bbox_min and bbox_max")
    if cfg.task and cfg.targets_json is None:
        task = cfg.task
    return task, targets


def _sample_one_handle(
    bbox_min: tuple[float, float, float] | list[float] | np.ndarray,
    bbox_max: tuple[float, float, float] | list[float] | np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    return sample_targets_in_bbox(bbox_min, bbox_max, 1, rng)[0]


def generate_reach_ik_dataset(config: ReachIkGenConfig, *, log: LogFn = print) -> Path:
    """Generate a LeRobot dataset from reach-IK trajectories. Returns output_root."""
    if config.episodes_per_target < 1:
        raise ValueError(f"episodes_per_target must be >= 1, got {config.episodes_per_target}")
    if config.pose_jitter_std < 0.0:
        raise ValueError(f"pose_jitter_std must be >= 0, got {config.pose_jitter_std}")

    master_rng = np.random.default_rng(config.seed)
    task_desc, fixed_targets = _resolve_targets(config, master_rng)
    bbox_mode = fixed_targets is None
    if bbox_mode:
        log(
            f"bbox mode: will keep sampling until {config.num_targets} successful "
            f"targets (grasp-only) task={task_desc!r}"
        )
    else:
        log(f"targets={len(fixed_targets)} task={task_desc!r} (grasp-only)")

    output_root = config.output_root
    if output_root is None:
        if config.targets_json is not None:
            stem = Path(config.targets_json).stem
            output_root = Path(config.targets_json).parent / f"unoarm_reach_ik_{stem}"
        else:
            output_root = Path(__file__).resolve().parents[1] / "data" / "unoarm_reach_ik"
    output_root = Path(output_root)

    if output_root.exists():
        if not config.overwrite:
            raise FileExistsError(f"{output_root} already exists. Pass overwrite=True to replace it.")
        log(f"Removing existing dataset at {output_root}")
        shutil.rmtree(output_root)

    from gym_unoarm.env import UnoarmEnv

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset.create(
        repo_id=config.repo_id,
        fps=FPS,
        features=dataset_features(),
        root=output_root,
        robot_type="unoarm",
        use_videos=True,
    )

    env = UnoarmEnv(
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
        max_episode_steps=10_000,
        scene=SCENE_REACH_SWORD,
        remove_shield=True,
        enable_execution_point=False,
        terminate_on_success=False,
    )

    warned_black = False
    written = 0
    skipped = 0
    success_targets = 0
    episode_metas: list[dict[str, Any]] = []

    def try_solve(handle: np.ndarray) -> tuple[list[np.ndarray] | None, np.ndarray | None]:
        poses_local: list[np.ndarray] | None = None
        home_local: np.ndarray | None = None
        for _attempt in range(max(1, int(config.ik_retries))):
            env.reset()
            place_sword_handle(env, handle)
            home_local = env._current_control_qpos().astype(np.float32).copy()
            poses_local = build_reach_ik_poses(
                env,
                handle,
                approach_offset_m=config.approach_offset_m,
            )
            if poses_local is not None:
                return poses_local, home_local
        return None, None

    def write_target(ti: int, handle: np.ndarray, poses: list[np.ndarray], home_q: np.ndarray) -> None:
        nonlocal written, warned_black
        for ep in range(config.episodes_per_target):
            rng = np.random.default_rng(config.seed + 1000 * ti + ep)
            env.reset()
            place_sword_handle(env, handle)
            start_q = home_q if home_q is not None else env._current_control_qpos().astype(np.float32)

            if config.pose_jitter_std > 0.0:
                jitter = rng.normal(0.0, config.pose_jitter_std, size=16).astype(np.float32)
                jitter[RIGHT_GRIPPER_INDEX] = 0.0
                jitter[7] = 0.0  # left gripper
                ep_poses = [clip_raw_pose(env, pose + jitter) for pose in poses]
                start_q = clip_raw_pose(env, start_q + jitter)
            else:
                ep_poses = poses

            raw_actions = build_episode_actions(
                ep_poses,
                config.segment_steps,
                config.hold_steps,
                rng,
                midpoint_noise_std=0.0,
                hold_noise_std=0.0,
                start=start_q,
            )

            last_info: dict[str, Any] = {}
            for raw_action in raw_actions:
                action = env._normalize(clip_raw_pose(env, raw_action))
                obs, _reward, _terminated, _truncated, last_info = env.step(action)
                if not warned_black and images_are_black(obs):
                    log("warning: rendered images are all black; check MuJoCo offscreen rendering.")
                    warned_black = True
                add_frame(dataset, obs, action.copy(), task_desc)

            dataset.save_episode()
            written += 1
            episode_metas.append(
                {
                    "episode_index": written - 1,
                    "target_index": int(ti),
                    "handle": [float(x) for x in handle.tolist()],
                }
            )
            attached = bool(last_info.get("attached", False))
            log(
                f"saved episode {written} (target {ti}, ep {ep + 1}/"
                f"{config.episodes_per_target}): {len(raw_actions)} frames "
                f"attached={attached} handle={handle.tolist()}"
            )

    try:
        if bbox_mode:
            assert config.bbox_min is not None and config.bbox_max is not None
            goal = int(config.num_targets)
            # Cap redraws so unreachable AABB does not hang forever.
            max_draws = max(goal * 25, goal + 50)
            draw_i = 0
            while success_targets < goal and draw_i < max_draws:
                handle = _sample_one_handle(config.bbox_min, config.bbox_max, master_rng)
                draw_i += 1
                poses, home_q = try_solve(handle)
                if poses is None or home_q is None:
                    skipped += 1
                    log(
                        f"skip draw {draw_i}: IK failed handle={handle.tolist()} "
                        f"(success {success_targets}/{goal})"
                    )
                    continue
                write_target(success_targets, handle, poses, home_q)
                success_targets += 1
            if success_targets < goal:
                log(
                    f"warning: only got {success_targets}/{goal} successful targets "
                    f"after {draw_i} draws (skipped={skipped})"
                )
        else:
            assert fixed_targets is not None
            for ti, handle in enumerate(fixed_targets):
                poses, home_q = try_solve(handle)
                if poses is None or home_q is None:
                    skipped += 1
                    log(f"skip target {ti}: IK failed handle={handle.tolist()}")
                    continue
                write_target(ti, handle, poses, home_q)
                success_targets += 1
    finally:
        dataset.finalize()
        env.close()
        if written > 0:
            meta = {
                "kind": "reach_ik",
                "mode": "grasp_only",
                "task": task_desc,
                "episodes": episode_metas,
                "repo_id": config.repo_id,
                "success_targets": int(success_targets),
                "skipped_targets": int(skipped),
            }
            meta_path = output_root / REACH_IK_META_NAME
            meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            log(f"wrote {meta_path.name}")
        log(
            f"dataset finalized at {output_root} written={written} "
            f"success_targets={success_targets} skipped_targets={skipped}"
        )

    if written == 0:
        raise RuntimeError("No episodes written (all targets failed IK or empty target list)")
    return Path(output_root)


def load_reach_ik_meta(root: Path) -> dict[str, Any] | None:
    path = Path(root) / REACH_IK_META_NAME
    if not path.is_file():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, dict) else None


def list_reach_ik_datasets(data_root: Path | None = None) -> list[dict[str, Any]]:
    """List local LeRobot roots that look like reach-IK (or any unoarm) datasets."""
    base = Path(data_root) if data_root is not None else DATA_ROOT
    if not base.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue
        meta_info = child / "meta" / "info.json"
        reach_meta = child / REACH_IK_META_NAME
        table_meta = child / TABLE_PLACE_IK_META_NAME
        if not meta_info.is_file() and not reach_meta.is_file() and not table_meta.is_file():
            # Prefer dirs that look like datasets
            if not (child / "data").is_dir() and not (child / "meta").is_dir():
                continue
        n_episodes = None
        repo_id = None
        kind = "unknown"
        if table_meta.is_file():
            kind = "table_place_ik"
            try:
                m = json.loads(table_meta.read_text(encoding="utf-8"))
                if isinstance(m, dict):
                    repo_id = m.get("repo_id")
                    if m.get("num_episodes_written") is not None:
                        n_episodes = int(m["num_episodes_written"])
                    eps = m.get("episodes")
                    if n_episodes is None and isinstance(eps, list):
                        n_episodes = len(eps)
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                pass
        elif reach_meta.is_file():
            kind = "reach_ik"
            try:
                m = json.loads(reach_meta.read_text(encoding="utf-8"))
                if isinstance(m, dict):
                    repo_id = m.get("repo_id")
                    eps = m.get("episodes")
                    if isinstance(eps, list):
                        n_episodes = len(eps)
            except (OSError, json.JSONDecodeError):
                pass
        if meta_info.is_file():
            try:
                info = json.loads(meta_info.read_text(encoding="utf-8"))
                if isinstance(info, dict):
                    if n_episodes is None:
                        n_episodes = int(info.get("total_episodes") or info.get("episodes") or 0) or None
                    repo_id = repo_id or info.get("repo_id")
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                pass
        # Prefer reach_ik / table_place named dirs or those with our meta
        name = child.name
        if kind == "unknown" and "reach_ik" not in name and "table_place" not in name:
            if not name.startswith("unoarm_"):
                continue
        if kind == "unknown":
            if reach_meta.is_file() or "reach_ik" in name:
                kind = "reach_ik"
            elif table_meta.is_file() or "table_place" in name:
                kind = "table_place_ik"
            else:
                kind = "unoarm"
        out.append(
            {
                "root": str(child),
                "name": name,
                "repo_id": repo_id or f"local/{name}",
                "n_episodes": n_episodes,
                "kind": kind,
            }
        )
    return out


def dataset_episode_count(root: Path, repo_id: str | None = None) -> int:
    table_meta = load_table_place_ik_meta(root)
    if table_meta and isinstance(table_meta.get("episodes"), list):
        return len(table_meta["episodes"])
    if table_meta and table_meta.get("num_episodes_written") is not None:
        return int(table_meta["num_episodes_written"])
    meta = load_reach_ik_meta(root)
    if meta and isinstance(meta.get("episodes"), list):
        return len(meta["episodes"])
    info_path = Path(root) / "meta" / "info.json"
    if info_path.is_file():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        if isinstance(info, dict):
            total = info.get("total_episodes")
            if total is not None:
                return int(total)
    # Fallback: open dataset
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    rid = repo_id or f"local/{Path(root).name}"
    ds = LeRobotDataset(rid, root=Path(root), download_videos=False, video_backend="pyav")
    return int(getattr(ds.meta, "total_episodes", 0) or 0)
