"""Webcam or recorded-clip frame source."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from so101_aimbot.config import REPO_ROOT, load_config


class Camera:
    """Open a webcam index or video clip based on `configs/camera.yaml`."""

    def __init__(
        self,
        index: int | None = None,
        width: int | None = None,
        height: int | None = None,
        *,
        source: str | None = None,
        clip_path: str | Path | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        cfg = config if config is not None else load_config("camera")

        source = source if source is not None else str(cfg.get("source", "webcam"))
        width = width if width is not None else int(cfg.get("width", 640))
        height = height if height is not None else int(cfg.get("height", 480))
        index = index if index is not None else int(cfg.get("index", 0))

        if source == "clip":
            path = Path(clip_path) if clip_path is not None else Path(str(cfg["clip_path"]))
            if not path.is_absolute():
                path = REPO_ROOT / path
            self.cap = cv2.VideoCapture(str(path))
            self._label = str(path)
        elif source == "webcam":
            self.cap = cv2.VideoCapture(index)
            self._label = f"webcam:{index}"
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        else:
            raise ValueError(f"Unknown camera source: {source!r} (expected 'webcam' or 'clip')")

        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera source {self._label}")

        self.width = width
        self.height = height
        self.source = source

    def read(self) -> NDArray[np.uint8]:
        success, frame = self.cap.read()
        if not success or frame is None:
            if self.source == "clip":
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                success, frame = self.cap.read()
            if not success or frame is None:
                raise RuntimeError(f"Could not read frame from {self._label}")
        return frame

    def read_latest(self, *, drain: int = 8) -> NDArray[np.uint8]:
        """Return the newest frame, discarding buffered ones first.

        Webcam drivers queue several frames. After a long blocking move/click,
        a plain ``read()`` can return an image captured mid-motion. Drain the
        queue so the next shot is planned from the post-click game state.
        """
        if self.source == "webcam":
            for _ in range(max(0, drain)):
                if not self.cap.grab():
                    break
        return self.read()

    def release(self) -> None:
        self.cap.release()

    def __enter__(self) -> Camera:
        return self

    def __exit__(self, *args: object) -> None:
        self.release()
