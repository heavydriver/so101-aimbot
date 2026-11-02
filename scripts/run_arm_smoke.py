"""Smoke-test pad controller: short move, oversize move (reset), click."""

from __future__ import annotations

from so101_aimbot.control import ArmController, PadMouseController
from so101_aimbot.config import load_config


def main() -> None:
    cfg = load_config("robot")
    arm = ArmController(cfg)
    pad = PadMouseController(arm, config=cfg)
    pad.connect(warm_up_ik=True)
    try:
        arm.home()
        print(
            f"Connected (dry_run={arm.dry_run}) "
            f"home_pad_xy={pad.current_pad_xy_mm()} "
            f"max_radius_mm={pad.max_radius_mm}"
        )

        print("Short move_polar(10 mm, 0 deg)…")
        pad.move_polar(50.0, 90.0)
        print(f"  pad_xy={pad.current_pad_xy_mm()}")
        
        pad.move_polar(20.0, -40.0)
        print(f"  pad_xy={pad.current_pad_xy_mm()}")
        
        pad.move_polar(20.0, 180.0)
        print(f"  pad_xy={pad.current_pad_xy_mm()}")
        
        # pad.move_xy_mm(0.0, 100.0)
        
        # arm.home()

        # oversize = pad.max_radius_mm * 2.5
        # print(f"Oversize move_polar({oversize:.1f} mm, 45 deg) with lift-home-continue…")
        # pad.move_polar(oversize, 45.0)
        # print(f"  pad_xy={pad.current_pad_xy_mm()}")

        print("click()…")
        pad.click()
        print("Arm / pad smoke test OK.")
    finally:
        arm.disconnect()


if __name__ == "__main__":
    main()
