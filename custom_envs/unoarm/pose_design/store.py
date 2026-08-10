"""Filesystem persistence for design and poses JSON files."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .export import to_poses_payload
from .models import DesignProject, slugify_name, validate_project


@dataclass
class DesignListItem:
    name: str
    task: str
    design_path: str
    poses_path: str
    keyframe_count: int
    playlist_length: int


class DesignStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def design_path(self, name: str) -> Path:
        return self.root / f"{slugify_name(name)}.design.json"

    def poses_path(self, name: str) -> Path:
        return self.root / f"{slugify_name(name)}.poses.json"

    def list(self) -> list[DesignListItem]:
        items: list[DesignListItem] = []
        for path in sorted(self.root.glob("*.design.json")):
            try:
                project = self.load(path.name.removesuffix(".design.json"))
            except (OSError, json.JSONDecodeError, ValueError):
                continue
            items.append(
                DesignListItem(
                    name=project.name,
                    task=project.task,
                    design_path=str(self.design_path(project.name)),
                    poses_path=str(self.poses_path(project.name)),
                    keyframe_count=len(project.keyframes),
                    playlist_length=len(project.playlist),
                )
            )
        return items

    def save(self, project: DesignProject) -> dict[str, str]:
        validate_project(project)
        project.name = slugify_name(project.name)
        design_file = self.design_path(project.name)
        poses_file = self.poses_path(project.name)
        design_file.write_text(
            json.dumps(project.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        poses_payload = to_poses_payload(project)
        poses_file.write_text(
            json.dumps(poses_payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return {"design_path": str(design_file), "poses_path": str(poses_file)}

    def load(self, name: str) -> DesignProject:
        path = self.design_path(name)
        if not path.exists():
            raise FileNotFoundError(f"Design not found: {path}")
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        project = DesignProject.from_dict(data)
        project.name = slugify_name(name)
        return project

    def delete(self, name: str) -> None:
        design = self.design_path(name)
        poses = self.poses_path(name)
        if not design.exists() and not poses.exists():
            raise FileNotFoundError(f"Design not found: {slugify_name(name)}")
        if design.exists():
            design.unlink()
        if poses.exists():
            poses.unlink()
