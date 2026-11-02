"""Teach pad home from the current arm pose (robot-only, no camera)."""

from __future__ import annotations

import argparse

from so101_aimbot.planning.calibrate import (
    load_calibration,
    set_max_radius_mm,
    teach_pad_home,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--teach-home",
        action="store_true",
        help="Capture home / contact Z / tip orientation from the live arm pose.",
    )
    parser.add_argument(
        "--max-radius-mm",
        type=float,
        default=None,
        help="Write max_radius_mm into configs/robot.yaml.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="With --teach-home, print taught values without writing robot.yaml.",
    )
    args = parser.parse_args()

    if args.max_radius_mm is not None:
        set_max_radius_mm(args.max_radius_mm)
        print(f"Updated max_radius_mm={args.max_radius_mm}")

    if args.teach_home:
        taught = teach_pad_home(save=not args.no_save)
        print("Taught pad home:")
        for key, value in taught.items():
            print(f"  {key}: {value}")
        return

    if args.max_radius_mm is None:
        cal = load_calibration()
        print(
            "Mouse↔cursor calibration:\n"
            f"  px_per_mm_x={cal.px_per_mm_x}\n"
            f"  px_per_mm_y={cal.px_per_mm_y}\n"
            "Pad teach: uv run python scripts/run_calibration.py --teach-home\n"
            "Radius:     uv run python scripts/run_calibration.py --max-radius-mm 40"
        )


if __name__ == "__main__":
    main()
