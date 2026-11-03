from __future__ import annotations

import numpy as np

from so101_aimbot.planning.ruckig_traj import JointLimits, run_joint_trajectory


def test_joint_trajectory_finishes_at_target() -> None:
    samples: list[list[float]] = []
    target = [5.0, -3.0, 2.0, 1.0]
    limits = JointLimits(
        velocity_deg_s=[20.0] * 4,
        acceleration_deg_s2=[100.0] * 4,
        jerk_deg_s3=[500.0] * 4,
    )

    run_joint_trajectory(
        dt_s=0.01,
        current_deg=[0.0] * 4,
        target_deg=target,
        limits=limits,
        on_sample=samples.append,
        realtime=False,
    )

    assert samples
    assert np.allclose(samples[-1], target)
    assert max(abs(sample[0]) for sample in samples) <= target[0] + 1e-9
