"""Evaluate a trained ACT policy on table_place pick-and-place.

Loads a checkpoint, rolls out in ``scene=table_place`` with peg XY taken from
the training meta (or sampled in the workspace), and reports success when the
peg ends inside the target circle (``info["success"]``).

Usage::

    uv run python custom_envs/unoarm/scripts/05_eval_table_place.py \
        --checkpoint outputs/train/.../checkpoints/010000/pretrained_model \
        --episodes 10
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lerobot.configs import PreTrainedConfig  # noqa: E402
from lerobot.policies import make_pre_post_processors  # noqa: E402
from lerobot.policies.act import ACTPolicy  # noqa: E402
from lerobot.policies.utils import build_inference_frame  # noqa: E402

from gym_unoarm.constants import (  # noqa: E402
    FPS,
    JOINTS,
    SCENE_TABLE_PLACE,
    is_peg_xy_in_workspace,
    sample_peg_xy,
)
from gym_unoarm.env import UnoarmEnv  # noqa: E402

MAX_STEPS = 400  # training episodes ~220–330 frames
# Default: draw eval peg XY from unique targets in the training meta.
DEFAULT_FROM_META = ROOT / "data" / "unoarm_test" / "table_place_ik_meta.json"
DEFAULT_N_ACTION_STEPS = 50  # match training chunk_size


def dataset_features() -> dict:
    image_feature = {
        "shape": (480, 640, 3),
        "dtype": "video",
        "names": ["height", "width", "channels"],
        "info": {"is_depth_map": False},
    }
    return {
        "action": {"shape": (16,), "dtype": "float32", "names": list(JOINTS)},
        "observation.state": {"shape": (16,), "dtype": "float32", "names": list(JOINTS)},
        "observation.images.top": image_feature,
        "observation.images.left_wrist": copy.deepcopy(image_feature),
        "observation.images.right_wrist": copy.deepcopy(image_feature),
    }


def obs_to_frame(obs: dict, device: torch.device) -> dict:
    state_np = np.asarray(obs["agent_pos"], dtype=np.float32)
    lerobot_obs: dict = {name: float(state_np[i]) for i, name in enumerate(JOINTS)}
    lerobot_obs["top"] = np.asarray(obs["pixels"]["top"])
    lerobot_obs["left_wrist"] = np.asarray(obs["pixels"]["left_wrist"])
    lerobot_obs["right_wrist"] = np.asarray(obs["pixels"]["right_wrist"])
    return build_inference_frame(
        observation=lerobot_obs,
        ds_features=dataset_features(),
        device=device,
        robot_type="unoarm",
    )


def unique_pegs_from_meta(meta_path: Path) -> list[tuple[float, float]]:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    seen: set[tuple[float, float]] = set()
    pegs: list[tuple[float, float]] = []
    for e in meta.get("episodes") or []:
        raw = e.get("peg_xy")
        if not raw or len(raw) < 2:
            continue
        key = (round(float(raw[0]), 6), round(float(raw[1]), 6))
        if key in seen:
            continue
        seen.add(key)
        pegs.append((float(raw[0]), float(raw[1])))
    return pegs


def run_eval(
    checkpoint: Path,
    episodes: int,
    seed: int,
    from_meta: Path | None,
    n_action_steps: int,
    max_steps: int,
) -> dict:
    config = PreTrainedConfig.from_pretrained(str(checkpoint))
    chunk = int(getattr(config, "chunk_size", 50))
    n_action_steps = int(n_action_steps)
    if n_action_steps < 1 or n_action_steps > chunk:
        raise ValueError(f"n_action_steps must be in [1, {chunk}], got {n_action_steps}")
    config.n_action_steps = n_action_steps
    print(f"[info] inference n_action_steps={n_action_steps} (chunk_size={chunk})")

    policy = ACTPolicy.from_pretrained(str(checkpoint), config=config)
    policy.eval()
    policy.config.n_action_steps = n_action_steps
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = policy.to(device)

    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    env = UnoarmEnv(
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
        scene=SCENE_TABLE_PLACE,
        terminate_on_success=False,
        max_episode_steps=max_steps + 10,
    )

    rng = np.random.default_rng(seed)
    if from_meta is not None:
        pegs = unique_pegs_from_meta(from_meta)
        rng.shuffle(pegs)
        targets = pegs[:episodes]
        if len(targets) < episodes:
            print(f"[warn] only {len(targets)} unique pegs in meta, requested {episodes}")
        print(f"[from_meta] {len(targets)} peg targets from {from_meta}")
        target_source = "from_meta"
    else:
        targets = [sample_peg_xy(rng) for _ in range(episodes)]
        target_source = "sample_workspace"

    successes = 0
    attach_final = 0
    per_episode: list[dict] = []

    for ep, peg_xy in enumerate(targets):
        peg_xy = (float(peg_xy[0]), float(peg_xy[1]))
        if not is_peg_xy_in_workspace(peg_xy):
            print(f"[warn] peg_xy={peg_xy} outside workspace; still evaluating")
        env.reset(options={"peg_xy": peg_xy})
        policy.reset()

        terminated = truncated = False
        step = 0
        saw_attach = False
        attach_step: int | None = None
        best_place = float("inf")
        last_info: dict = {}
        t0 = time.time()
        while not (terminated or truncated) and step < max_steps:
            obs_frame = obs_to_frame(env._get_obs(), device)
            processed = preprocess(obs_frame)
            with torch.inference_mode():
                action = policy.select_action(processed)
            action = postprocess(action)
            action_np = (
                action.squeeze(0).detach().cpu().numpy()
                if isinstance(action, torch.Tensor)
                else np.asarray(action).squeeze(0)
            )
            action_np = np.asarray(action_np, dtype=np.float32).reshape(-1)[:16]
            _obs, _reward, terminated, truncated, last_info = env.step(action_np)
            if bool(last_info.get("peg_attached", False)):
                saw_attach = True
                if attach_step is None:
                    attach_step = step
            pd = last_info.get("place_distance")
            if pd is not None:
                best_place = min(best_place, float(pd))
            step += 1

        success = bool(last_info.get("success", False))
        final_attached = bool(last_info.get("peg_attached", False))
        place_dist = float(last_info.get("place_distance") or best_place)
        successes += int(success)
        attach_final += int(final_attached)
        elapsed = time.time() - t0
        tag = "OK" if success else "FAIL"
        print(
            f"[{tag}] ep{ep + 1:3d}/{len(targets)} peg=[{peg_xy[0]:+.3f},{peg_xy[1]:+.3f}] "
            f"place={place_dist * 100:.1f}cm attach={int(saw_attach)}@{attach_step} "
            f"final_attach={int(final_attached)} steps={step} ({elapsed:.1f}s)",
            flush=True,
        )
        per_episode.append(
            {
                "episode": ep,
                "peg_xy": list(peg_xy),
                "success": success,
                "saw_attach": saw_attach,
                "attach_step": attach_step,
                "final_attached": final_attached,
                "place_distance_m": place_dist,
                "place_distance_cm": place_dist * 100,
                "steps": step,
                "elapsed_s": elapsed,
            }
        )

    env.close()
    n = len(targets)
    summary = {
        "checkpoint": str(checkpoint.resolve()),
        "seed": seed,
        "target_source": target_source,
        "from_meta_path": str(from_meta) if from_meta else None,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "scene": SCENE_TABLE_PLACE,
        "fps": FPS,
        "success_criterion": "info.success (peg in target circle)",
        "n_action_steps": n_action_steps,
        "max_steps": max_steps,
        "episodes": n,
        "successes": successes,
        "success_rate": successes / n if n else 0.0,
        "final_attach_rate": attach_final / n if n else 0.0,
        "attach_ever_rate": float(np.mean([e["saw_attach"] for e in per_episode])) if n else 0.0,
        "mean_place_distance_cm": float(np.mean([e["place_distance_cm"] for e in per_episode]))
        if n
        else None,
        "episodes_detail": per_episode,
    }

    ckpt_dir = checkpoint.resolve()
    run_root = ckpt_dir.parents[2]
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = run_root / "eval" / f"table_place_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"\n=== table_place eval ===\n"
        f"success: {successes}/{n} ({100 * summary['success_rate']:.1f}%)\n"
        f"ever_attach: {100 * summary['attach_ever_rate']:.1f}%  "
        f"final_attach: {100 * summary['final_attach_rate']:.1f}%\n"
        f"mean place_dist: {summary['mean_place_distance_cm']:.1f} cm\n"
        f"wrote {out_dir}",
        flush=True,
    )
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to pretrained_model directory.",
    )
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--from-meta",
        type=Path,
        default=DEFAULT_FROM_META,
        help="table_place_ik_meta.json — sample unique training peg XY (default: unoarm_test).",
    )
    p.add_argument(
        "--no-from-meta",
        action="store_true",
        help="Ignore training meta; sample peg XY randomly in the workspace.",
    )
    p.add_argument("--n-action-steps", type=int, default=DEFAULT_N_ACTION_STEPS)
    p.add_argument("--max-steps", type=int, default=MAX_STEPS)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_dir():
        raise FileNotFoundError(args.checkpoint)
    from_meta = None if args.no_from_meta else args.from_meta
    if from_meta is not None and not from_meta.is_file():
        raise FileNotFoundError(f"--from-meta not found: {from_meta}")
    run_eval(
        checkpoint=args.checkpoint,
        episodes=args.episodes,
        seed=args.seed,
        from_meta=from_meta,
        n_action_steps=args.n_action_steps,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
