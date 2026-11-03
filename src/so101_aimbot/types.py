"""Shared contracts between perception, planning, and control."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class Target:
    """A detected circular target in ROI/screen pixel coordinates."""

    x: float
    y: float
    confidence: float = 1.0
    radius_px: float = 0.0


@dataclass(frozen=True, slots=True)
class CursorState:
    """Detected mouse cursor position in ROI/screen pixel coordinates."""

    x: float
    y: float


@dataclass(slots=True)
class FrameBundle:
    """Raw camera frame plus optional warped game ROI."""

    raw: NDArray[np.uint8]
    warped: NDArray[np.uint8] | None = None


@dataclass(frozen=True, slots=True)
class MouseCommand:
    """Relative mouse motion on the pad, plus optional click."""

    dx: float
    dy: float
    click: bool = False


@dataclass(slots=True)
class PerceptionResult:
    """One perception tick: targets, cursor, and selected goal."""

    targets: list[Target] = field(default_factory=list)
    cursor: CursorState | None = None
    selected: Target | None = None
