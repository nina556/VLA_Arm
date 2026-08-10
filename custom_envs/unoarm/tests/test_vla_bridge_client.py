from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from webapp.vla_bridge_client import (
    BridgeConfig,
    VlaBridgeClient,
    build_joint_chunk_body,
    denormalize_actions,
)


def test_denormalize_midpoint_is_zero_when_symmetric() -> None:
    limits = np.array([[-1.0, 1.0]] * 16, dtype=np.float32)
    out = denormalize_actions(np.zeros((2, 16), dtype=np.float32), limits)
    assert out.shape == (2, 16)
    np.testing.assert_allclose(out, 0.0, atol=1e-6)


def test_denormalize_extremes() -> None:
    limits = np.zeros((16, 2), dtype=np.float32)
    limits[:, 0] = -2.0
    limits[:, 1] = 4.0
    actions = np.full((1, 16), -1.0, dtype=np.float32)
    actions[0, 3] = 1.0
    out = denormalize_actions(actions, limits)
    assert out[0, 0] == pytest.approx(-2.0)
    assert out[0, 3] == pytest.approx(4.0)


def test_build_joint_chunk_body_shape_and_meta() -> None:
    actions = [[0.0] * 8 + [0.01] + [0.0] * 7]
    body = build_joint_chunk_body(
        actions,
        arms="right",
        fps=20,
        execute=False,
        instruction="Right Reverse Rolling Arm",
    )
    assert body["command_type"] == "joint_chunk"
    assert body["unit"] == "rad"
    assert body["arms"] == "right"
    assert body["execute"] is False
    assert len(body["actions"]) == 1
    assert len(body["actions"][0]) == 16
    assert body["meta"]["instruction"] == "Right Reverse Rolling Arm"


def test_bridge_config_rejects_bad_arms() -> None:
    with pytest.raises(ValueError, match="arms"):
        BridgeConfig(arms="middle").normalized()


def test_client_posts_json_payload() -> None:
    client = VlaBridgeClient(BridgeConfig(enabled=True, base_url="http://192.168.10.38:8765", execute=False))
    fake_resp = MagicMock()
    fake_resp.status = 200
    fake_resp.read.return_value = json.dumps({"ok": True, "points": 1}).encode("utf-8")
    fake_resp.__enter__.return_value = fake_resp
    fake_resp.__exit__.return_value = False

    with patch("urllib.request.urlopen", return_value=fake_resp) as mocked:
        result = client.post_joint_chunk(np.zeros((1, 16)), instruction="smoke")
    assert result["ok"] is True
    req = mocked.call_args.args[0]
    assert req.full_url.endswith("/api/mujoco/vla_joint_chunk")
    payload = json.loads(req.data.decode("utf-8"))
    assert payload["actions"][0][0] == 0.0
    assert payload["meta"]["instruction"] == "smoke"
