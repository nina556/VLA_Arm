"""Data models for Unoarm pose design projects."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from gym_unoarm.constants import CONTROL_JOINTS

_SLUG_RE = re.compile(r"[^A-Za-z0-9_-]+")


def slugify_name(name: str) -> str:
    cleaned = _SLUG_RE.sub("_", name.strip()).strip("_-")
    if not cleaned:
        raise ValueError("name must contain at least one alphanumeric character")
    return cleaned


def pose_dict_from_vector(values: list[float] | tuple[float, ...]) -> dict[str, float]:
    if len(values) != len(CONTROL_JOINTS):
        raise ValueError(f"pose must have {len(CONTROL_JOINTS)} values, got {len(values)}")
    return {name: float(values[i]) for i, name in enumerate(CONTROL_JOINTS)}


def vector_from_pose_dict(pose: dict[str, Any]) -> list[float]:
    if not isinstance(pose, dict):
        raise ValueError(f"pose must be an object, got {type(pose)}")
    unknown = set(pose.keys()) - set(CONTROL_JOINTS)
    if unknown:
        raise ValueError(f"unknown joint names: {sorted(unknown)}")
    return [float(pose.get(name, 0.0)) for name in CONTROL_JOINTS]


def normalize_pose_dict(pose: dict[str, Any]) -> dict[str, float]:
    return pose_dict_from_vector(vector_from_pose_dict(pose))


@dataclass
class DesignProject:
    name: str
    task: str = ""
    keyframes: dict[str, dict[str, float]] = field(default_factory=dict)
    playlist: list[str] = field(default_factory=list)
    version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "name": self.name,
            "task": self.task,
            "keyframes": {k: dict(v) for k, v in self.keyframes.items()},
            "playlist": list(self.playlist),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DesignProject:
        if not isinstance(data, dict):
            raise ValueError("design project must be a JSON object")
        name = str(data.get("name", "")).strip()
        if not name:
            raise ValueError("design project requires a non-empty 'name'")
        keyframes_raw = data.get("keyframes", {})
        if not isinstance(keyframes_raw, dict):
            raise ValueError("'keyframes' must be an object")
        keyframes = {str(k): normalize_pose_dict(v) for k, v in keyframes_raw.items()}
        playlist_raw = data.get("playlist", [])
        if not isinstance(playlist_raw, list):
            raise ValueError("'playlist' must be a list")
        playlist = [str(item) for item in playlist_raw]
        return cls(
            version=int(data.get("version", 1)),
            name=name,
            task=str(data.get("task", "")),
            keyframes=keyframes,
            playlist=playlist,
        )


def validate_playlist_structure(project: DesignProject) -> None:
    """Structural checks for preview/export: keyframes + playlist refs."""
    if not project.keyframes:
        raise ValueError("keyframes must contain at least one keyframe")
    if not project.playlist:
        raise ValueError("playlist must contain at least one entry")
    for i, key in enumerate(project.playlist):
        if key not in project.keyframes:
            raise ValueError(f"playlist[{i}] references unknown keyframe {key!r}")
    for name, pose in project.keyframes.items():
        normalize_pose_dict(pose)
        if not str(name).strip():
            raise ValueError("keyframe names must be non-empty")


def validate_project(project: DesignProject) -> None:
    if not project.name.strip():
        raise ValueError("name must be non-empty")
    slugify_name(project.name)
    if not str(project.task).strip():
        raise ValueError("task must be a non-empty English instruction")
    validate_playlist_structure(project)
