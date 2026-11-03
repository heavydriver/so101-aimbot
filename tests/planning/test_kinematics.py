from __future__ import annotations

import numpy as np

from so101_aimbot.config import load_config
from so101_aimbot.planning.kinematics import ACTIVE_JOINTS, So101Kinematics


def _kinematics() -> So101Kinematics:
    config = load_config("robot")
    return So101Kinematics(
        config["urdf_path"],
        config["home_arm_deg"],
        target_frame=config["target_frame"],
    )


def test_reduced_model_contains_only_active_joints() -> None:
    kinematics = _kinematics()
    assert kinematics.model.nq == 4
    assert set(kinematics.model.names[1:]) == set(ACTIVE_JOINTS)


def test_fk_ik_reaches_nearby_target() -> None:
    kinematics = _kinematics()
    target = kinematics.home_pose.position_mm + np.array([5.0, 0.0, 0.0])
    solution = kinematics.inverse(
        target,
        initial_deg=kinematics.home_active_deg,
        tolerance_mm=0.5,
    )
    reached = kinematics.forward(solution).position_mm

    assert np.linalg.norm(reached - target) <= 0.5
    assert np.all(solution >= kinematics.lower_deg)
    assert np.all(solution <= kinematics.upper_deg)
