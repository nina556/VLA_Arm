"""Expand a design project playlist into flat poses payloads."""

from __future__ import annotations

from typing import Any

from .models import DesignProject, normalize_pose_dict, validate_playlist_structure


def expand_playlist(project: DesignProject) -> list[dict[str, float]]:
    validate_playlist_structure(project)
    return [normalize_pose_dict(project.keyframes[name]) for name in project.playlist]


def to_poses_payload(project: DesignProject) -> dict[str, Any]:
    from .models import validate_project

    validate_project(project)
    return {
        "task": project.task.strip(),
        "poses": expand_playlist(project),
    }
