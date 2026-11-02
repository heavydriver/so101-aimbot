"""Turn perception output into a mouse motion / click command."""

from __future__ import annotations

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
    *,
    click_threshold_px: float = 8.0,
) -> MouseCommand | None:
    """Plan a relative mouse move (and optional click) toward `target`."""
    if cursor is None or target is None:
        return None

    dx_px = target.x - cursor.x
    dy_px = target.y - cursor.y
    dx_mm, dy_mm = pixel_delta_to_mouse_mm(dx_px, dy_px)
    distance = (dx_px**2 + dy_px**2) ** 0.5
    return MouseCommand(dx=dx_mm, dy=dy_mm, click=distance <= click_threshold_px)
