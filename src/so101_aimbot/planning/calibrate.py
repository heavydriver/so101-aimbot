"""Calibrate physical mouse motion ↔ on-screen cursor motion."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MouseCursorCalibration:
    """Linear map from mouse pad Δmm to cursor Δpx."""

    px_per_mm_x: float = 1.0
    px_per_mm_y: float = 1.0


def load_calibration() -> MouseCursorCalibration:
    """Load a saved calibration (stub returns identity)."""
    return MouseCursorCalibration()


def pixel_delta_to_mouse_mm(
    dx_px: float,
    dy_px: float,
    calibration: MouseCursorCalibration | None = None,
) -> tuple[float, float]:
    """Convert a desired cursor pixel delta into mouse pad millimetres."""
    cal = calibration or load_calibration()
    return dx_px / cal.px_per_mm_x, dy_px / cal.px_per_mm_y
