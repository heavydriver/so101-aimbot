"""Turn perception output into a mouse motion / click command."""

from __future__ import annotations

import math

from so101_aimbot.planning.calibrate import pixel_delta_to_mouse_mm
from so101_aimbot.planning.pad_geometry import polar_to_xy_mm, plan_polar_segments
from so101_aimbot.types import CursorState, MouseCommand, Target

__all__ = [
    "plan_mouse_command",
    "polar_to_xy_mm",
    "plan_polar_segments",
]


def plan_mouse_command(
    cursor: CursorState | None,
    target: Target | None,
) -> MouseCommand | None:
    """Move toward `target`; click only when the observed cursor is inside it."""
    if cursor is None or target is None:
        return None

    dx_px = target.x - cursor.x
    dy_px = target.y - cursor.y
    dx_mm, dy_mm = pixel_delta_to_mouse_mm(dx_px, dy_px)
    cursor_is_inside = math.hypot(dx_px, dy_px) <= target.radius_px
    return MouseCommand(dx=dx_mm, dy=dy_mm, click=cursor_is_inside)
