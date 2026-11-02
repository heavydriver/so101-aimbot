"""SO-101 forward / inverse kinematics via PyRoki."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as np
import pyroki as pk
import yourdfpy
from numpy.typing import NDArray

from so101_aimbot.config import REPO_ROOT

# PyRoki actuated order for so101_new_calib.urdf (topo-sorted).
PYROKI_ACTUATED_NAMES = (
    "gripper",
    "wrist_roll",
    "wrist_flex",
    "elbow_flex",
    "shoulder_lift",
    "shoulder_pan",
)

# LeRobot / project joint order.
PROJECT_JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


def _project_to_pyroki(q_project: NDArray[np.floating]) -> NDArray[np.float64]:
    idx = {name: i for i, name in enumerate(PROJECT_JOINT_NAMES)}
    return np.array([q_project[idx[name]] for name in PYROKI_ACTUATED_NAMES], dtype=np.float64)


def _pyroki_to_project(q_pyroki: NDArray[np.floating]) -> NDArray[np.float64]:
    idx = {name: i for i, name in enumerate(PYROKI_ACTUATED_NAMES)}
    return np.array([q_pyroki[idx[name]] for name in PROJECT_JOINT_NAMES], dtype=np.float64)


@dataclass(slots=True)
class EndEffectorPose:
    """Tip pose in robot base frame."""

    position_m: NDArray[np.float64]  # (3,)
    wxyz: NDArray[np.float64]  # (4,)

    @property
    def xyz(self) -> NDArray[np.float64]:
        return self.position_m


class So101Kinematics:
    """PyRoki FK/IK wrapper for the SO-101."""

    def __init__(
        self,
        urdf_path: str | Path,
        *,
        target_link: str = "gripper_frame_link",
    ) -> None:
        path = Path(urdf_path)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if not path.is_file():
            raise FileNotFoundError(f"URDF not found: {path}")

        urdf = yourdfpy.URDF.load(
            str(path),
            load_meshes=False,
            build_scene_graph=True,
        )
        self._robot = pk.Robot.from_urdf(urdf)
        self._target_link = target_link
        if target_link not in self._robot.links.names:
            raise ValueError(
                f"target_link {target_link!r} not in {self._robot.links.names}"
            )
        self._target_link_index = self._robot.links.names.index(target_link)
        self._ik_compiled = False

    @property
    def robot(self) -> pk.Robot:
        return self._robot

    @property
    def target_link(self) -> str:
        return self._target_link

    def forward_kinematics(self, joints_rad_project: Sequence[float]) -> EndEffectorPose:
        """Map project-order joint radians → tip pose (wxyz + xyz metres)."""
        q = np.asarray(joints_rad_project, dtype=np.float64).reshape(6)
        q_pk = _project_to_pyroki(q)
        link_poses = np.asarray(self._robot.forward_kinematics(q_pk))
        pose7 = link_poses[self._target_link_index]
        return EndEffectorPose(
            position_m=np.asarray(pose7[4:7], dtype=np.float64).copy(),
            wxyz=np.asarray(pose7[0:4], dtype=np.float64).copy(),
        )

    def inverse_kinematics(
        self,
        position_m: Sequence[float],
        wxyz: Sequence[float],
        *,
        guess_rad_project: Sequence[float] | None = None,
        pos_weight: float = 50.0,
        ori_weight: float = 10.0,
        rest_weight: float = 1.0,
    ) -> list[float]:
        """Solve IK for tip pose; returns project-order joint radians (6)."""
        target_pos = np.asarray(position_m, dtype=np.float64).reshape(3)
        target_wxyz = np.asarray(wxyz, dtype=np.float64).reshape(4)
        if guess_rad_project is None:
            mid = 0.5 * (
                np.asarray(self._robot.joints.lower_limits)
                + np.asarray(self._robot.joints.upper_limits)
            )
            guess_pk = np.asarray(mid, dtype=np.float64)
        else:
            guess_pk = _project_to_pyroki(
                np.asarray(guess_rad_project, dtype=np.float64).reshape(6)
            )

        cfg_pk = _solve_ik(
            self._robot,
            self._target_link_index,
            target_wxyz,
            target_pos,
            guess_pk,
            pos_weight=pos_weight,
            ori_weight=ori_weight,
            rest_weight=rest_weight,
        )
        self._ik_compiled = True
        return _pyroki_to_project(cfg_pk).tolist()

    def warm_up(self, guess_rad_project: Sequence[float] | None = None) -> None:
        """Trigger JAX JIT compile with a throwaway IK solve."""
        pose = self.forward_kinematics(
            guess_rad_project
            if guess_rad_project is not None
            else [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        )
        self.inverse_kinematics(
            pose.position_m,
            pose.wxyz,
            guess_rad_project=guess_rad_project,
        )


def inverse_kinematics(target_xyz: Sequence[float]) -> list[float]:
    """Backward-compatible stub API: position-only IK with identity orientation."""
    kin = So101Kinematics(REPO_ROOT / "assets" / "so101" / "so101_new_calib.urdf")
    return kin.inverse_kinematics(target_xyz, [1.0, 0.0, 0.0, 0.0])


@jdc.jit
def _solve_ik_jax(
    robot: pk.Robot,
    target_link_index: jnp.ndarray,
    target_wxyz: jnp.ndarray,
    target_position: jnp.ndarray,
    rest_cfg: jnp.ndarray,
    pos_weight: float,
    ori_weight: float,
    rest_weight: float,
) -> jnp.ndarray:
    joint_var = robot.joint_var_cls(0)
    costs = [
        pk.costs.pose_cost_analytic_jac(
            robot,
            joint_var,
            jaxlie.SE3.from_rotation_and_translation(
                jaxlie.SO3(target_wxyz), target_position
            ),
            target_link_index,
            pos_weight=pos_weight,
            ori_weight=ori_weight,
        ),
        pk.costs.limit_cost(
            robot,
            joint_var,
            weight=100.0,
        ),
        pk.costs.rest_cost(
            joint_var,
            rest_cfg,
            weight=rest_weight,
        ),
    ]
    sol = (
        jaxls.LeastSquaresProblem(costs=costs, variables=[joint_var])
        .analyze()
        .solve(
            initial_vals=jaxls.VarValues.make([joint_var.with_value(rest_cfg)]),
            verbose=False,
            linear_solver="dense_cholesky",
            trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
        )
    )
    return sol[joint_var]


def _solve_ik(
    robot: pk.Robot,
    target_link_index: int,
    target_wxyz: NDArray[np.floating],
    target_position: NDArray[np.floating],
    rest_cfg: NDArray[np.floating],
    *,
    pos_weight: float,
    ori_weight: float,
    rest_weight: float,
) -> NDArray[np.float64]:
    cfg = _solve_ik_jax(
        robot,
        jnp.array(target_link_index),
        jnp.asarray(target_wxyz, dtype=jnp.float32),
        jnp.asarray(target_position, dtype=jnp.float32),
        jnp.asarray(rest_cfg, dtype=jnp.float32),
        float(pos_weight),
        float(ori_weight),
        float(rest_weight),
    )
    return np.asarray(cfg, dtype=np.float64)
