from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from webapp.modes import AppMode, ModeController, ModeError  # noqa: E402


def test_mode_switch_chat_design() -> None:
    ctl = ModeController()
    assert ctl.mode == AppMode.CHAT
    ctl.set_mode("design")
    assert ctl.mode == AppMode.DESIGN
    ctl.set_mode(AppMode.CHAT)
    assert ctl.mode == AppMode.CHAT


def test_require_mode_blocks_chat_ops_in_design() -> None:
    ctl = ModeController()
    ctl.set_mode(AppMode.DESIGN)
    try:
        ctl.require_mode(AppMode.CHAT)
        raise AssertionError("expected ModeError")
    except ModeError as exc:
        assert exc.status_code == 409


def test_generating_locks_mode_switch() -> None:
    ctl = ModeController()
    ctl.set_mode(AppMode.DESIGN)
    ctl.enter_generating()
    assert ctl.mode == AppMode.GENERATING
    try:
        ctl.set_mode(AppMode.CHAT)
        raise AssertionError("expected ModeError")
    except ModeError:
        pass
    ctl.leave_generating(resume=AppMode.DESIGN)
    assert ctl.mode == AppMode.DESIGN
