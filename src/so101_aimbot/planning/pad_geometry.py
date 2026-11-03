"""Pad-frame geometry: polar commands and max-radius disk segmentation."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray


def polar_to_xy_mm(distance_mm: float, direction_deg: float) -> tuple[float, float]:
    """Convert polar pad command to Cartesian millimetres.

    Convention: ``0°`` = +pad X, angles increase counter-clockwise when viewed
    from above.
    """
    theta = math.radians(direction_deg)
    return float(distance_mm * math.cos(theta)), float(distance_mm * math.sin(theta))


def xy_to_polar_mm(dx_mm: float, dy_mm: float) -> tuple[float, float]:
    """Inverse of :func:`polar_to_xy_mm` → ``(distance_mm, direction_deg)``."""
    distance = math.hypot(dx_mm, dy_mm)
    direction = math.degrees(math.atan2(dy_mm, dx_mm))
    return float(distance), float(direction)


def ray_to_disk_boundary_mm(
    pos_xy_mm: tuple[float, float] | NDArray[np.floating],
    direction_deg: float,
    max_radius_mm: float,
    *,
    eps: float = 1e-6,
) -> float:
    """Distance along a ray from ``pos`` to the disk boundary of radius ``R``.

    The disk is centered at the pad origin (home). Returns ``0`` if the point is
    already outside / on the boundary in that direction (no positive progress).
    """
    x, y = float(pos_xy_mm[0]), float(pos_xy_mm[1])
    theta = math.radians(direction_deg)
    dx, dy = math.cos(theta), math.sin(theta)
    # Solve ||p + t d||^2 = R^2 for smallest t >= 0.
    # t^2 + 2(p·d)t + (||p||^2 - R^2) = 0
    r2 = float(max_radius_mm) ** 2
    pd = x * dx + y * dy
    pp = x * x + y * y
    disc = pd * pd - (pp - r2)
    if disc < 0.0:
        return 0.0
    sqrt_disc = math.sqrt(disc)
    t1 = -pd - sqrt_disc
    t2 = -pd + sqrt_disc
    candidates = [t for t in (t1, t2) if t > eps]
    if not candidates:
        return 0.0
    return float(min(candidates))


def plan_polar_segments(
    distance_mm: float,
    direction_deg: float,
    *,
    start_xy_mm: tuple[float, float] = (0.0, 0.0),
    max_radius_mm: float,
    eps: float = 1e-3,
    max_segments: int = 64,
) -> list[float]:
    """Split a polar move into on-pad segment lengths (mm) with home resets implied.

    Each returned length is a contact slide from the current pad pose. Between
    segments the caller must lift → home → drop (except after the last segment).
    Simulation assumes each reset returns the tip to the pad origin ``(0, 0)``.
    """
    if distance_mm <= eps:
        return []
    if max_radius_mm <= 0.0:
        raise ValueError("max_radius_mm must be positive")

    remaining = float(distance_mm)
    pos = (float(start_xy_mm[0]), float(start_xy_mm[1]))
    segments: list[float] = []

    for _ in range(max_segments):
        if remaining <= eps:
            break
        step = min(
            remaining,
            ray_to_disk_boundary_mm(pos, direction_deg, max_radius_mm, eps=eps),
        )
        if step <= eps:
            # Already at boundary in this direction — reset to home and retry.
            pos = (0.0, 0.0)
            step = min(
                remaining,
                ray_to_disk_boundary_mm(pos, direction_deg, max_radius_mm, eps=eps),
            )
            if step <= eps:
                break
            segments.append(step)
            remaining -= step
            # After a slide from home, next iteration continues from new pos unless
            # more distance remains (then caller resets again).
            theta = math.radians(direction_deg)
            pos = (step * math.cos(theta), step * math.sin(theta))
            if remaining > eps:
                pos = (0.0, 0.0)
            continue

        segments.append(step)
        remaining -= step
        theta = math.radians(direction_deg)
        pos = (pos[0] + step * math.cos(theta), pos[1] + step * math.sin(theta))
        if remaining > eps:
            pos = (0.0, 0.0)

    if remaining > eps:
        raise RuntimeError(
            f"Could not fully segment move; {remaining:.3f} mm left "
            f"(max_radius_mm={max_radius_mm})"
        )
    return segments
