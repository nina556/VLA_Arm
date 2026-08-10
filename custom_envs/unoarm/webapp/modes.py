"""Exclusive app mode state machine for chat / design / generating."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AppMode(str, Enum):
    CHAT = "chat"
    DESIGN = "design"
    GENERATING = "generating"


class ModeError(RuntimeError):
    """Raised when an operation is not allowed in the current mode."""

    def __init__(self, message: str, *, status_code: int = 409) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass
class ModeController:
    _mode: AppMode = AppMode.CHAT

    @property
    def mode(self) -> AppMode:
        return self._mode

    def set_mode(self, mode: AppMode | str) -> AppMode:
        target = AppMode(mode) if not isinstance(mode, AppMode) else mode
        if target == AppMode.GENERATING:
            raise ModeError("Cannot switch directly to generating; start a generate job instead.")
        if self._mode == AppMode.GENERATING and target != AppMode.GENERATING:
            raise ModeError("Cannot switch mode while dataset generation is running.")
        self._mode = target
        return self._mode

    def enter_generating(self) -> None:
        if self._mode == AppMode.GENERATING:
            raise ModeError("A generate job is already running.")
        self._mode = AppMode.GENERATING

    def leave_generating(self, *, resume: AppMode = AppMode.DESIGN) -> None:
        if self._mode != AppMode.GENERATING:
            return
        if resume == AppMode.GENERATING:
            raise ModeError("resume mode cannot be generating")
        self._mode = resume

    def require_mode(self, *allowed: AppMode) -> None:
        if self._mode not in allowed:
            names = ", ".join(m.value for m in allowed)
            raise ModeError(
                f"Operation not allowed in mode {self._mode.value!r}; requires one of: {names}"
            )
