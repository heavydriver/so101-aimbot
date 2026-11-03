"""Unit tests for pad polar math and disk segmentation."""

from __future__ import annotations

import math

import pytest

from so101_aimbot.planning.pad_geometry import (
    plan_polar_segments,
    polar_to_xy_mm,
    ray_to_disk_boundary_mm,
    xy_to_polar_mm,
)


def test_polar_roundtrip() -> None:
    dx, dy = polar_to_xy_mm(25.0, 30.0)
    d, th = xy_to_polar_mm(dx, dy)
    assert d == pytest.approx(25.0)
    assert th == pytest.approx(30.0)


def test_ray_from_origin_is_radius() -> None:
    assert ray_to_disk_boundary_mm((0.0, 0.0), 0.0, 40.0) == pytest.approx(40.0)
    assert ray_to_disk_boundary_mm((0.0, 0.0), 90.0, 40.0) == pytest.approx(40.0)


def test_ray_from_midpoint() -> None:
    # At x=20 on +X, remaining to R=40 is 20.
    assert ray_to_disk_boundary_mm((20.0, 0.0), 0.0, 40.0) == pytest.approx(20.0)
    # Opposite direction toward -X: can go to -40 → 60 mm.
    assert ray_to_disk_boundary_mm((20.0, 0.0), 180.0, 40.0) == pytest.approx(60.0)


def test_ray_at_boundary_forward_is_zero() -> None:
    assert ray_to_disk_boundary_mm((40.0, 0.0), 0.0, 40.0) == pytest.approx(0.0)


def test_plan_segments_single() -> None:
    segs = plan_polar_segments(15.0, 0.0, max_radius_mm=40.0)
    assert segs == pytest.approx([15.0])


def test_plan_segments_multiple_resets() -> None:
    segs = plan_polar_segments(100.0, 0.0, max_radius_mm=40.0)
    assert sum(segs) == pytest.approx(100.0)
    assert all(s <= 40.0 + 1e-9 for s in segs)
    assert len(segs) >= 3


def test_plan_segments_from_offset_start() -> None:
    segs = plan_polar_segments(
        50.0, 0.0, start_xy_mm=(30.0, 0.0), max_radius_mm=40.0
    )
    # First leg only 10 mm to the boundary, then reset and continue.
    assert segs[0] == pytest.approx(10.0)
    assert sum(segs) == pytest.approx(50.0)


def test_polar_axes() -> None:
    assert polar_to_xy_mm(10.0, 0.0) == pytest.approx((10.0, 0.0))
    x, y = polar_to_xy_mm(10.0, 90.0)
    assert x == pytest.approx(0.0, abs=1e-9)
    assert y == pytest.approx(10.0)
    assert math.hypot(*polar_to_xy_mm(10.0, 45.0)) == pytest.approx(10.0)
