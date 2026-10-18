"""Debug viewer for the screen camera / clip (perception only)."""

from __future__ import annotations

import cv2
import numpy as np

from so101_aimbot.perception import Camera
from so101_aimbot.pipeline import Pipeline
from so101_aimbot.types import PerceptionResult


def main() -> None:
    pipeline = Pipeline(actuate=False)
    cam = Camera()
    try:
        while True:
            frame = cam.read()
            bundle, result = pipeline.perceive(frame)

            raw = frame.copy()
            if pipeline._corners is not None:
                quad = pipeline._corners.astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(raw, [quad], True, (0, 255, 0), 2)

            view = bundle.warped if bundle.warped is not None else raw
            overlay = _draw_result(view, result)
            cv2.imshow("camera", raw)
            cv2.imshow("perception", overlay)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()


def _draw_result(frame: np.ndarray, result: PerceptionResult) -> np.ndarray:
    view = frame.copy()
    for target in result.targets:
        center = (int(round(target.x)), int(round(target.y)))
        color = (0, 255, 255) if result.selected is target else (0, 255, 0)
        cv2.circle(view, center, 14, color, 2)
    if result.cursor is not None:
        cursor = (int(round(result.cursor.x)), int(round(result.cursor.y)))
        cv2.drawMarker(view, cursor, (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
    label = f"targets {len(result.targets)}"
    if result.cursor is not None:
        label += f"  cursor ({result.cursor.x:.0f}, {result.cursor.y:.0f})"
    cv2.putText(view, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return view


if __name__ == "__main__":
    main()
