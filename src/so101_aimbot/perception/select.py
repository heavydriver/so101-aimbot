"""Choose which detected target to aim at."""

from __future__ import annotations

import math

from so101_aimbot.types import CursorState, Target


def select_nearest_target(
    targets: list[Target],
    cursor: CursorState | None,
) -> Target | None:
    """Return the target closest to `cursor`, or None if there are no targets."""
    if not targets:
        return None
    if cursor is None:
        return targets[0]

    def distance(t: Target) -> float:
        return math.hypot(t.x - cursor.x, t.y - cursor.y)

    return min(targets, key=distance)
