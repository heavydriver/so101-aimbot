"""Crop / warp the Gridshot game region from a camera frame."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from so101_aimbot.config import load_config


def find_game_corners(
    frame: NDArray[np.uint8],
    *,
    config: dict[str, Any] | None = None,
) -> NDArray[np.float32] | None:
    """Locate the four corners of the game display in `frame`.

    The monitor is the large dark quadrilateral against a brighter surround.
    Returns corners ordered TL, TR, BR, BL, or None when the screen is not visible.
    """
    cfg = config if config is not None else load_config("perception")
    manual = cfg.get("game_corners")
    if manual is not None:
        corners = np.asarray(manual, dtype=np.float32)
        if corners.shape != (4, 2):
            raise ValueError("game_corners must be four [x, y] points (TL, TR, BR, BL)")
        return corners

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    screen_max = int(cfg.get("screen_max_value", 130))
    mask = np.where(blur < screen_max, np.uint8(255), np.uint8(0))

    height, width = mask.shape
    kernel = _odd_kernel(min(height, width) * 0.02)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((kernel, kernel), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    min_area = float(cfg.get("screen_min_area_ratio", 0.25)) * height * width
    if cv2.contourArea(contour) < min_area:
        return None

    corners = _corners_from_contour(contour)
    if corners is None or cv2.contourArea(corners.reshape(-1, 1, 2)) < min_area:
        return None
    return corners


def warp_game_region(
    frame: NDArray[np.uint8],
    corners: NDArray[np.float32],
    size: tuple[int, int] = (960, 540),
) -> NDArray[np.uint8]:
    """Perspective-warp the game region defined by `corners` to `size` (w, h)."""
    width, height = size
    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    return cv2.warpPerspective(frame, matrix, (width, height))


def _odd_kernel(value: float) -> int:
    kernel = max(5, int(round(value)))
    return kernel if kernel % 2 == 1 else kernel + 1


def _corners_from_contour(contour: NDArray[np.int32]) -> NDArray[np.float32] | None:
    """Fit a TL, TR, BR, BL quad to the convex hull of `contour`."""
    hull = cv2.convexHull(contour)
    peri = cv2.arcLength(hull, True)
    for fraction in (0.02, 0.04, 0.06, 0.08):
        approx = cv2.approxPolyDP(hull, fraction * peri, True)
        if len(approx) == 4:
            return _order_corners(approx.reshape(4, 2))

    points = hull.reshape(-1, 2)
    if len(points) < 4:
        return None
    return _order_corners(points)


def _order_corners(points: NDArray[np.generic]) -> NDArray[np.float32]:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    total = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    ordered = np.array(
        [
            pts[np.argmin(total)],  # TL
            pts[np.argmin(diff)],  # TR (max x - y)
            pts[np.argmax(total)],  # BR
            pts[np.argmax(diff)],  # BL
        ],
        dtype=np.float32,
    )
    return ordered
