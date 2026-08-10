"""Simple linear interpolation preview trajectories (no noise)."""

from __future__ import annotations

import numpy as np

from .export import expand_playlist
from .models import DesignProject, vector_from_pose_dict


def build_preview_trajectory(project: DesignProject, segment_steps: int = 20) -> list[np.ndarray]:
    """Build a raw-qpos trajectory by linearly interpolating playlist poses.

    Does not wrap with RAW_ZERO and does not apply midpoint noise. The first
    keyframe is included as the starting pose; each subsequent segment contributes
    ``segment_steps`` frames ending on the next keyframe.
    """
    if segment_steps < 1:
        raise ValueError(f"segment_steps must be >= 1, got {segment_steps}")
    poses = [np.asarray(vector_from_pose_dict(p), dtype=np.float32) for p in expand_playlist(project)]
    if len(poses) == 1:
        return [poses[0].copy()]

    trajectory: list[np.ndarray] = [poses[0].copy()]
    for start, target in zip(poses, poses[1:], strict=False):
        for alpha in np.linspace(1.0 / segment_steps, 1.0, segment_steps, dtype=np.float32):
            frame = ((1.0 - alpha) * start + alpha * target).astype(np.float32)
            trajectory.append(frame)
    return trajectory
