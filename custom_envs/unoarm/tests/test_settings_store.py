"""Tests for web settings store."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from webapp.settings_store import (  # noqa: E402
    API_KEY_MASK,
    load_settings,
    merge_settings_patch,
    public_settings,
    save_settings,
)


def test_save_load_roundtrip(tmp_path: Path):
    path = tmp_path / "web_settings.json"
    data = load_settings(path)
    data["checkpoint"] = "/tmp/ckpt"
    data["api_key"] = "sk-secret"
    data["allowed_tasks"] = ["Reach the sword handle"]
    save_settings(data, path)
    loaded = load_settings(path)
    assert loaded["checkpoint"] == "/tmp/ckpt"
    assert loaded["api_key"] == "sk-secret"
    assert loaded["allowed_tasks"] == ["Reach the sword handle"]


def test_public_settings_masks_key():
    pub = public_settings({"api_key": "sk-secret", "checkpoint": "x"})
    assert pub["api_key"] == API_KEY_MASK
    assert pub["api_key_set"] is True


def test_merge_keeps_key_when_masked():
    current = {"api_key": "sk-old", "checkpoint": "a"}
    merged = merge_settings_patch(
        current,
        {"api_key": API_KEY_MASK, "checkpoint": "b"},
    )
    assert merged["api_key"] == "sk-old"
    assert merged["checkpoint"] == "b"
