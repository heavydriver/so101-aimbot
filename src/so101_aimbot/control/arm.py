"""SO-101 arm interface (stub)."""

from __future__ import annotations

from typing import Sequence


class ArmController:
    """Send joint or Cartesian commands to the SO-101."""

    def __init__(self) -> None:
        self._connected = False

    def connect(self) -> None:
        """Open the serial / SDK connection (stub)."""
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def move_joints(self, joints: Sequence[float]) -> None:
        """Command joint angles in radians (stub)."""
        if not self._connected:
            raise RuntimeError("Arm is not connected")
        _ = joints

    def home(self) -> None:
        """Move to the configured home pose (stub)."""
        if not self._connected:
            raise RuntimeError("Arm is not connected")
