import cv2
import numpy as np

from so101_aimbot.perception.detect import detect_cursor, detect_targets
from so101_aimbot.perception.preprocess import find_game_corners, warp_game_region


def test_find_game_corners_on_dark_screen() -> None:
    frame = np.full((300, 400, 3), 240, dtype=np.uint8)
    cv2.rectangle(frame, (40, 30), (360, 260), (15, 20, 25), thickness=-1)

    corners = find_game_corners(frame, config={"screen_max_value": 130, "screen_min_area_ratio": 0.2})
    assert corners is not None
    assert corners.shape == (4, 2)
    expected = np.array([[40, 30], [360, 30], [360, 260], [40, 260]], dtype=np.float32)
    assert np.allclose(corners, expected, atol=12)


def test_warp_maps_corners_to_output_edges() -> None:
    frame = np.zeros((100, 120, 3), dtype=np.uint8)
    frame[20:80, 10:110] = (0, 0, 255)
    corners = np.array([[10, 20], [109, 20], [109, 79], [10, 79]], dtype=np.float32)

    warped = warp_game_region(frame, corners, size=(80, 40))
    assert warped.shape == (40, 80, 3)
    assert warped[20, 40, 2] > 200


def test_detect_targets_finds_three_magenta_circles() -> None:
    frame = np.full((540, 960, 3), (40, 30, 20), dtype=np.uint8)
    magenta = tuple(int(c) for c in cv2.cvtColor(np.uint8([[[150, 200, 220]]]), cv2.COLOR_HSV2BGR)[0, 0])
    centers = [(300, 220), (480, 260), (640, 300)]
    for center in centers:
        cv2.circle(frame, center, 18, magenta, thickness=-1)

    targets = detect_targets(frame, config=_perception_config())
    assert len(targets) == 3
    found = {(round(t.x), round(t.y)) for t in targets}
    for x, y in centers:
        assert any(abs(fx - x) <= 4 and abs(fy - y) <= 4 for fx, fy in found)


def test_detect_targets_ignores_small_noise() -> None:
    frame = np.full((540, 960, 3), (40, 30, 20), dtype=np.uint8)
    magenta = tuple(int(c) for c in cv2.cvtColor(np.uint8([[[150, 180, 200]]]), cv2.COLOR_HSV2BGR)[0, 0])
    cv2.circle(frame, (100, 100), 2, magenta, thickness=-1)
    assert detect_targets(frame, config=_perception_config()) == []


def test_detect_cursor_finds_white_plus() -> None:
    frame = np.full((540, 960, 3), 90, dtype=np.uint8)
    cx, cy = 420, 250
    cv2.line(frame, (cx - 10, cy), (cx + 10, cy), (255, 255, 255), 3)
    cv2.line(frame, (cx, cy - 10), (cx, cy + 10), (255, 255, 255), 3)

    cursor = detect_cursor(frame, config=_perception_config())
    assert cursor is not None
    assert abs(cursor.x - cx) <= 3
    assert abs(cursor.y - cy) <= 3


def test_detect_targets_ignores_bright_menu() -> None:
    frame = np.full((540, 960, 3), 230, dtype=np.uint8)
    magenta = tuple(int(c) for c in cv2.cvtColor(np.uint8([[[150, 200, 220]]]), cv2.COLOR_HSV2BGR)[0, 0])
    cv2.circle(frame, (480, 270), 20, magenta, thickness=-1)
    assert detect_targets(frame, config=_perception_config()) == []


def test_detect_cursor_ignores_round_highlight() -> None:
    frame = np.full((540, 960, 3), 90, dtype=np.uint8)
    cv2.circle(frame, (480, 270), 8, (255, 255, 255), thickness=-1)
    assert detect_cursor(frame, config=_perception_config()) is None


def _perception_config() -> dict:
    return {
        "reference_height": 540,
        "target_hsv_lower": [128, 40, 70],
        "target_hsv_upper": [172, 255, 255],
        "blur_ksize": 5,
        "morph_ksize": 5,
        "close_ksize": 11,
        "max_scene_value": 165,
        "min_target_radius_px": 7.0,
        "max_target_radius_px": 28.0,
        "min_peak_separation_px": 18.0,
        "border_margin_px": 8,
        "max_targets": 6,
        "cursor_sat_max": 50,
        "cursor_whiteness_min": 140,
        "cursor_shape_min": 60,
        "cursor_compact_min": 80,
        "cursor_score_min": 11000,
        "cursor_top_ignore_px": 40,
    }
