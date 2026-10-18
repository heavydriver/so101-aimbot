"""Detect purple Gridshot targets and the mouse crosshair."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from so101_aimbot.config import load_config
from so101_aimbot.types import CursorState, Target


def detect_targets(
    frame: NDArray[np.uint8],
    *,
    config: dict[str, Any] | None = None,
) -> list[Target]:
    """Find bright purple target circles in the (warped) game frame.

    Magenta pixels are segmented in HSV, then each blob center is a local peak
    of the distance transform so touching spheres stay separate.
    """
    cfg = config if config is not None else load_config("perception")
    height, width = frame.shape[:2]
    if height < 16 or width < 16:
        return []

    scale = height / float(cfg.get("reference_height", 540))
    blur_ksize = _odd_at_least(int(cfg.get("blur_ksize", 5)), 3)
    open_ksize = _odd_at_least(int(cfg.get("morph_ksize", 5)), 3)
    close_ksize = _odd_at_least(int(cfg.get("close_ksize", 11)), 3)
    min_radius = float(cfg.get("min_target_radius_px", 7.0)) * scale
    max_radius = float(cfg.get("max_target_radius_px", 28.0)) * scale
    separation = float(cfg.get("min_peak_separation_px", 18.0)) * scale
    margin = float(cfg.get("border_margin_px", 8)) * scale
    max_targets = int(cfg.get("max_targets", 6))

    if not _scene_is_game(frame, cfg):
        return []

    blurred = cv2.GaussianBlur(frame, (blur_ksize, blur_ksize), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    lower = np.array(cfg.get("target_hsv_lower", [128, 40, 70]), dtype=np.uint8)
    upper = np.array(cfg.get("target_hsv_upper", [172, 255, 255]), dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    opened = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
    closed = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, opened)
    # Close fills the white crosshair when it sits on a sphere.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, closed)

    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    peak_window = _odd_at_least(int(round(11 * scale)), 5)
    dilated = cv2.dilate(distance, np.ones((peak_window, peak_window), np.uint8))
    peaks = ((distance >= dilated) & (distance >= min_radius)).astype(np.uint8)
    count, _labels, _stats, centroids = cv2.connectedComponentsWithStats(peaks)

    candidates: list[tuple[float, float, float]] = []
    for index in range(1, count):
        x, y = float(centroids[index, 0]), float(centroids[index, 1])
        if not (margin <= x < width - margin and margin <= y < height - margin):
            continue
        radius = _local_max(distance, int(round(x)), int(round(y)))
        if radius < min_radius or radius > max_radius:
            continue
        candidates.append((radius, x, y))

    candidates.sort(reverse=True)
    kept: list[tuple[float, float, float]] = []
    separation_sq = separation * separation
    for radius, x, y in candidates:
        if any((x - kx) ** 2 + (y - ky) ** 2 < separation_sq for _, kx, ky in kept):
            continue
        kept.append((radius, x, y))
        if len(kept) >= max_targets:
            break

    expected = max(max_radius, 1.0)
    return [
        Target(x=x, y=y, confidence=float(np.clip(radius / expected, 0.0, 1.0)))
        for radius, x, y in kept
    ]


def detect_cursor(
    frame: NDArray[np.uint8],
    *,
    config: dict[str, Any] | None = None,
) -> CursorState | None:
    """Estimate the crosshair position in the (warped) game frame.

    The reticle is a small unsaturated plus. Sphere highlights are bright too,
    but they are round and saturated, so a plus-shape score on a whiteness
    image rejects them.
    """
    cfg = config if config is not None else load_config("perception")
    if not _scene_is_game(frame, cfg):
        return None

    height, width = frame.shape[:2]
    scale = height / float(cfg.get("reference_height", 540))
    near = _scaled_radii((3, 4, 5, 6), scale, minimum=2)
    far = _scaled_radii((14, 18, 22), scale, minimum=near[-1] + 4)
    reach = far[-1]
    if height <= 2 * reach + 4 or width <= 2 * reach + 4:
        return None

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1].astype(np.float32)
    value = hsv[:, :, 2].astype(np.float32)
    sat_max = float(cfg.get("cursor_sat_max", 50))
    whiteness = value * np.clip((sat_max - saturation) / sat_max, 0.0, 1.0)

    shape, compact = _plus_response(whiteness, near, far)
    score = np.clip(shape, 0.0, None) * np.clip(compact, 0.0, None)
    center = whiteness[reach : height - reach, reach : width - reach]
    score[center < float(cfg.get("cursor_whiteness_min", 140))] = 0.0

    top_ignore = int(round(float(cfg.get("cursor_top_ignore_px", 40)) * scale))
    if top_ignore > reach:
        score[: top_ignore - reach, :] = 0.0

    shape_min = float(cfg.get("cursor_shape_min", 60))
    compact_min = float(cfg.get("cursor_compact_min", 80))
    score_min = float(cfg.get("cursor_score_min", 11000))
    valid = (shape >= shape_min) & (compact >= compact_min) & (score >= score_min)
    if not np.any(valid):
        return None

    masked = np.where(valid, score, 0.0)
    local_y, local_x = np.unravel_index(int(np.argmax(masked)), masked.shape)
    return CursorState(x=float(local_x + reach), y=float(local_y + reach))


def _scene_is_game(frame: NDArray[np.uint8], cfg: dict[str, Any]) -> bool:
    """Gridshot's view is a dark grid. The bright results page is not."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return float(np.median(hsv[:, :, 2])) <= float(cfg.get("max_scene_value", 165))


def _plus_response(
    channel: NDArray[np.floating[Any]],
    near: tuple[int, ...],
    far: tuple[int, ...],
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    """Return (axis-minus-diagonal, near-axis-minus-far-axis) on the interior."""
    reach = far[-1]
    near_axis = _sample_mean(channel, near, reach, diagonal=False)
    near_diag = _sample_mean(channel, near, reach, diagonal=True)
    far_axis = _sample_mean(channel, far, reach, diagonal=False)
    return near_axis - near_diag, near_axis - far_axis


def _sample_mean(
    channel: NDArray[np.floating[Any]],
    distances: tuple[int, ...],
    reach: int,
    *,
    diagonal: bool,
) -> NDArray[np.floating[Any]]:
    height, width = channel.shape
    total = np.zeros((height - 2 * reach, width - 2 * reach), dtype=np.float32)
    for distance in distances:
        if diagonal:
            samples = (
                channel[reach + distance : height - reach + distance, reach + distance : width - reach + distance],
                channel[reach + distance : height - reach + distance, reach - distance : width - reach - distance],
                channel[reach - distance : height - reach - distance, reach + distance : width - reach + distance],
                channel[reach - distance : height - reach - distance, reach - distance : width - reach - distance],
            )
        else:
            samples = (
                channel[reach: height - reach, reach + distance : width - reach + distance],
                channel[reach: height - reach, reach - distance : width - reach - distance],
                channel[reach + distance : height - reach + distance, reach: width - reach],
                channel[reach - distance : height - reach - distance, reach: width - reach],
            )
        total += samples[0] + samples[1] + samples[2] + samples[3]
    return total / (4 * len(distances))


def _scaled_radii(distances: tuple[int, ...], scale: float, *, minimum: int) -> tuple[int, ...]:
    scaled: list[int] = []
    floor = minimum
    for distance in distances:
        value = max(floor, int(round(distance * scale)))
        if not scaled or value > scaled[-1]:
            scaled.append(value)
            floor = value + 1
    return tuple(scaled)


def _local_max(distance: NDArray[np.floating[Any]], x: int, y: int) -> float:
    y0, y1 = max(0, y - 3), min(distance.shape[0], y + 4)
    x0, x1 = max(0, x - 3), min(distance.shape[1], x + 4)
    return float(distance[y0:y1, x0:x1].max())


def _odd_at_least(value: int, minimum: int) -> int:
    value = max(minimum, value)
    return value if value % 2 == 1 else value + 1
