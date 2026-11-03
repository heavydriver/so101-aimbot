"""Forward and inverse kinematics for the four moving SO-101 arm joints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pinocchio as pin
from numpy.typing import NDArray

from so101_aimbot.config import REPO_ROOT

ACTIVE_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
)
ARM_JOINTS = ACTIVE_JOINTS + ("wrist_roll",)
LOCKED_JOINTS = ("wrist_roll", "gripper")


@dataclass(frozen=True, slots=True)
class EndEffectorPose:
    """End-effector pose in the robot base frame."""

    position_mm: NDArray[np.float64]
    rotation: NDArray[np.float64]


class IKError(RuntimeError):
    """Raised when the requested end-effector position is unreachable."""


class So101Kinematics:
    """Small Pinocchio wrapper with wrist-roll and gripper locked."""

    def __init__(
        self,
        urdf_path: str | Path,
        home_arm_deg: Sequence[float],
        *,
        target_frame: str = "gripper_frame_link",
    ) -> None:
        path = Path(urdf_path)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if not path.is_file():
            raise FileNotFoundError(f"URDF not found: {path}")

        home = _vector(home_arm_deg, len(ARM_JOINTS), "home_arm_deg")
        full_model = pin.buildModelFromUrdf(str(path))
        reference = pin.neutral(full_model)
        for name, angle_deg in zip(ARM_JOINTS, home, strict=True):
            reference[_joint_q_index(full_model, name)] = np.deg2rad(angle_deg)

        locked_ids = [_joint_id(full_model, name) for name in LOCKED_JOINTS]
        self.model = pin.buildReducedModel(full_model, locked_ids, reference)
        self.data = self.model.createData()
        self._q_indices = [_joint_q_index(self.model, name) for name in ACTIVE_JOINTS]
        self._frame_id = self.model.getFrameId(target_frame)
        if self._frame_id >= self.model.nframes:
            raise ValueError(f"Unknown target frame: {target_frame!r}")

        self.lower_deg = np.rad2deg(self.model.lowerPositionLimit[self._q_indices])
        self.upper_deg = np.rad2deg(self.model.upperPositionLimit[self._q_indices])
        self.home_active_deg = home[: len(ACTIVE_JOINTS)].copy()
        self.home_pose = self.forward(self.home_active_deg)

    def forward(self, active_deg: Sequence[float]) -> EndEffectorPose:
        """Compute the end-effector pose from four joint angles in degrees."""
        q = self._configuration(active_deg)
        pin.framesForwardKinematics(self.model, self.data, q)
        pose = self.data.oMf[self._frame_id]
        return EndEffectorPose(
            position_mm=np.asarray(pose.translation * 1000.0, dtype=float).copy(),
            rotation=np.asarray(pose.rotation, dtype=float).copy(),
        )

    def inverse(
        self,
        position_mm: Sequence[float],
        *,
        initial_deg: Sequence[float],
        preferred_rotation: NDArray[np.float64] | None = None,
        tolerance_mm: float = 0.5,
        max_iterations: int = 200,
    ) -> list[float]:
        """Solve position IK, softly preferring the supplied orientation."""
        target = _vector(position_mm, 3, "position_mm") / 1000.0
        q = self._configuration(initial_deg)
        preferred = (
            self.home_pose.rotation
            if preferred_rotation is None
            else np.asarray(preferred_rotation, dtype=float).reshape(3, 3)
        )

        for _ in range(max_iterations):
            pin.framesForwardKinematics(self.model, self.data, q)
            pose = self.data.oMf[self._frame_id]
            position_error = target - pose.translation
            if np.linalg.norm(position_error) * 1000.0 <= tolerance_mm:
                return self._degrees(q)

            jacobian = pin.computeFrameJacobian(
                self.model,
                self.data,
                q,
                self._frame_id,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )
            rotation_error = pose.rotation @ pin.log3(pose.rotation.T @ preferred)

            # Millimetres make translation dominate this soft preference.
            orientation_weight = 0.1
            system = np.vstack(
                (jacobian[:3] * 1000.0, jacobian[3:] * orientation_weight)
            )
            error = np.concatenate(
                (position_error * 1000.0, rotation_error * orientation_weight)
            )
            damping = 1e-4
            step = system.T @ np.linalg.solve(
                system @ system.T + damping * np.eye(6), error
            )
            q = pin.integrate(self.model, q, 0.6 * step)
            q = np.clip(q, self.model.lowerPositionLimit, self.model.upperPositionLimit)

        final = self.forward(self._degrees(q)).position_mm
        final_error = np.linalg.norm(final - target * 1000.0)
        raise IKError(
            f"IK failed to reach target within {tolerance_mm:.2f} mm "
            f"(error={final_error:.2f} mm)"
        )

    def _configuration(self, active_deg: Sequence[float]) -> NDArray[np.float64]:
        values = _vector(active_deg, len(ACTIVE_JOINTS), "active_deg")
        q = pin.neutral(self.model)
        q[self._q_indices] = np.deg2rad(values)
        return q

    def _degrees(self, q: NDArray[np.floating]) -> list[float]:
        return np.rad2deg(np.asarray(q)[self._q_indices]).tolist()


def _joint_id(model: pin.Model, name: str) -> int:
    joint_id = model.getJointId(name)
    if joint_id == 0:
        raise ValueError(f"URDF is missing joint {name!r}")
    return joint_id


def _joint_q_index(model: pin.Model, name: str) -> int:
    joint = model.joints[_joint_id(model, name)]
    if joint.nq != 1:
        raise ValueError(f"Expected one-DOF joint {name!r}, got nq={joint.nq}")
    return joint.idx_q


def _vector(values: Sequence[float], size: int, name: str) -> NDArray[np.float64]:
    vector = np.asarray(values, dtype=float).reshape(-1)
    if vector.size != size:
        raise ValueError(f"{name} must contain {size} values")
    return vector
