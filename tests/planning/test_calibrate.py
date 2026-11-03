"""Unit tests for mouse-to-warped-cursor scaling."""

from __future__ import annotations

import pytest

from so101_aimbot.planning.calibrate import (
    MouseCursorCalibration,
    pixel_delta_to_mouse_mm,
    px_per_mm_warped,
)


def test_px_per_mm_warped_scales_to_perception_resolution() -> None:
    scale = px_per_mm_warped(
        dpi=800,
        sensitivity=1.0,
        game_pixels=1920,
        warp_pixels=960,
    )
    assert scale == pytest.approx((800 / 25.4) * 0.5)


def test_pixel_delta_to_mouse_mm_uses_axis_scales() -> None:
    calibration = MouseCursorCalibration(px_per_mm_x=20.0, px_per_mm_y=10.0)
    assert pixel_delta_to_mouse_mm(
        40.0, -20.0, calibration
    ) == pytest.approx((2.0, -2.0))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dpi", 0),
        ("sensitivity", 0),
        ("game_pixels", 0),
        ("warp_pixels", 0),
    ],
)
def test_px_per_mm_warped_rejects_non_positive_inputs(
    field: str, value: float
) -> None:
    kwargs: dict[str, float | int] = {
        "dpi": 800,
        "sensitivity": 1.0,
        "game_pixels": 1920,
        "warp_pixels": 960,
    }
    kwargs[field] = value
    with pytest.raises(ValueError):
        px_per_mm_warped(**kwargs)
