"""Calibrate mouse pad Δmm ↔ on-screen cursor Δpx (stub entrypoint)."""

from __future__ import annotations

from so101_aimbot.planning.calibrate import load_calibration


def main() -> None:
    cal = load_calibration()
    print(
        "Mouse↔cursor calibration (stub identity):\n"
        f"  px_per_mm_x={cal.px_per_mm_x}\n"
        f"  px_per_mm_y={cal.px_per_mm_y}\n"
        "Replace this script with the interactive calibration routine."
    )


if __name__ == "__main__":
    main()
