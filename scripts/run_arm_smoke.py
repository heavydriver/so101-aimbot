"""Smoke-test pad controller: short move, oversize move (reset), click."""

from __future__ import annotations

import argparse

from so101_aimbot.control import ArmController, MouseController
from so101_aimbot.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = load_config("robot")
    if args.dry_run:
        cfg.update(dry_run=True, port=None)
    arm = ArmController(cfg)
    mouse = MouseController(arm, config=cfg)
    mouse.connect()
    try:
        arm.home()
        print(
            f"Connected (dry_run={arm.dry_run}) "
            f"home_pad_xy={mouse.current_pad_xy_mm()} "
            f"max_radius_mm={mouse.max_radius_mm}"
        )
        
        # print("go_home() with lift → above home → drop…")
        # mouse.go_home()
        # print(f"  pad_xy={mouse.current_pad_xy_mm()}")

        print("Short move_polar(10 mm, 0 deg)…")
        mouse.move_polar(10.0, 180.0)
        print(f"  pad_xy={mouse.current_pad_xy_mm()}")
        
        mouse.move_polar(50.0, 90.0)
        print(f"  pad_xy={mouse.current_pad_xy_mm()}")
        
        mouse.move_polar(30.0, -45.0)
        print(f"  pad_xy={mouse.current_pad_xy_mm()}")

        mouse.move_polar(80.0, 230.0)
        print(f"  pad_xy={mouse.current_pad_xy_mm()}")
        
        # mouse.move_xy_mm(-50.0, -20.0)
        # print(f"  pad_xy={mouse.current_pad_xy_mm()}")

        print("click()…")
        mouse.click()
        print("Arm / pad smoke test OK.")
    finally:
        arm.disconnect()


if __name__ == "__main__":
    main()
