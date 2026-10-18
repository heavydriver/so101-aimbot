"""Mouse grip / contact helpers and command execution (stub)."""

from __future__ import annotations

from so101_aimbot.control.arm import ArmController
from so101_aimbot.types import MouseCommand


def execute_mouse_command(arm: ArmController, command: MouseCommand) -> None:
    """Execute a relative mouse move and optional click via the arm.

    Stub: map dx/dy into a Cartesian pad motion and trigger click actuation.
    """
    _ = arm
    _ = command
