from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_gen.scripted import load_poses_from_json  # noqa: E402
from gym_unoarm.constants import CONTROL_JOINTS  # noqa: E402
from pose_design.export import expand_playlist, to_poses_payload  # noqa: E402
from pose_design.models import DesignProject, slugify_name, validate_project, validate_playlist_structure  # noqa: E402
from pose_design.preview import build_preview_trajectory  # noqa: E402
from pose_design.store import DesignStore  # noqa: E402


def _full_pose(**overrides: float) -> dict[str, float]:
    pose = {name: 0.0 for name in CONTROL_JOINTS}
    pose.update(overrides)
    return pose


def test_slugify_name() -> None:
    assert slugify_name(" Hello World! ") == "Hello_World"
    with pytest.raises(ValueError):
        slugify_name("@@@")


def test_expand_playlist_allows_repeats() -> None:
    project = DesignProject(
        name="demo",
        task="Wave both arms.",
        keyframes={
            "a": _full_pose(Left_Joint1=1.0),
            "b": _full_pose(Right_Joint1=-1.0),
        },
        playlist=["a", "b", "a"],
    )
    poses = expand_playlist(project)
    assert len(poses) == 3
    assert poses[0]["Left_Joint1"] == 1.0
    assert poses[1]["Right_Joint1"] == -1.0
    assert poses[2]["Left_Joint1"] == 1.0


def test_validate_rejects_missing_playlist_ref() -> None:
    project = DesignProject(
        name="demo",
        task="Do a thing.",
        keyframes={"a": _full_pose()},
        playlist=["missing"],
    )
    with pytest.raises(ValueError, match="unknown keyframe"):
        validate_project(project)


def test_store_roundtrip_and_poses_compatible(tmp_path: Path) -> None:
    store = DesignStore(tmp_path)
    project = DesignProject(
        name="my_action",
        task="Raise the left arm.",
        keyframes={"up": _full_pose(Left_Joint1=0.7)},
        playlist=["up", "up"],
    )
    paths = store.save(project)
    loaded = store.load("my_action")
    assert loaded.task == project.task
    assert loaded.playlist == ["up", "up"]

    poses_path = Path(paths["poses_path"])
    assert poses_path.exists()
    task, poses = load_poses_from_json(poses_path)
    assert task == "Raise the left arm."
    assert len(poses) == 2
    assert poses[0][0] == pytest.approx(0.7)

    payload = json.loads(poses_path.read_text(encoding="utf-8"))
    assert payload == to_poses_payload(loaded)


def test_preview_trajectory_length() -> None:
    project = DesignProject(
        name="demo",
        task="Move.",
        keyframes={
            "a": _full_pose(Left_Joint1=0.0),
            "b": _full_pose(Left_Joint1=1.0),
        },
        playlist=["a", "b"],
    )
    traj = build_preview_trajectory(project, segment_steps=5)
    assert len(traj) == 1 + 5
    assert isinstance(traj[0], np.ndarray)
    assert traj[-1][0] == pytest.approx(1.0)


def test_preview_allows_empty_task() -> None:
    project = DesignProject(
        name="untitled",
        task="",
        keyframes={"a": _full_pose(Left_Joint1=0.2)},
        playlist=["a"],
    )
    validate_playlist_structure(project)
    with pytest.raises(ValueError, match="task"):
        validate_project(project)
    traj = build_preview_trajectory(project, segment_steps=3)
    assert len(traj) == 1
