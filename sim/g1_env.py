from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import mujoco
import numpy as np

from robostl.attacks.base import Attack, ObservationAttack, ObservationContext
from robostl.attacks.terrain import align_heightfield_baseline
from robostl.core.config import DeployConfig
from robostl.core.state import SimState
from robostl.metrics.stability import (
    calculate_stability_margin,
    get_support_polygon,
    resolve_ground_geom_ids,
)
from robostl.policies.torchscript import TorchScriptPolicy


def get_gravity_orientation(quaternion: np.ndarray) -> np.ndarray:
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3, dtype=np.float32)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def pd_control(
    target_q: np.ndarray,
    q: np.ndarray,
    kp: np.ndarray,
    target_dq: np.ndarray,
    dq: np.ndarray,
    kd: np.ndarray,
) -> np.ndarray:
    """Calculates torques from position commands."""
    return (target_q - q) * kp + (target_dq - dq) * kd


@dataclass
class ObservationBuffer:
    obs: np.ndarray


class G1ObservationBuilder:
    def __init__(
        self,
        config: DeployConfig,
        obs_attacks: Optional[list[ObservationAttack]] = None,
    ) -> None:
        self.config = config
        self.obs_buffer = ObservationBuffer(
            obs=np.zeros(self.config.num_obs, dtype=np.float32)
        )
        self.obs_attacks = obs_attacks or []
        self.context = ObservationContext(slices=self._build_slices())
        for attack in self.obs_attacks:
            attack.reset(self.context)

    def reset(self) -> None:
        for attack in self.obs_attacks:
            attack.reset(self.context)

    def _build_slices(self) -> dict[str, slice]:
        num_actions = self.config.num_actions
        base = 9
        return {
            "omega": slice(0, 3),
            "gravity": slice(3, 6),
            "cmd": slice(6, 9),
            "qj": slice(base, base + num_actions),
            "dqj": slice(base + num_actions, base + 2 * num_actions),
            "action": slice(base + 2 * num_actions, base + 3 * num_actions),
            "phase": slice(base + 3 * num_actions, base + 3 * num_actions + 2),
        }

    def build(
        self,
        data: mujoco.MjData,
        action: np.ndarray,
        cmd: np.ndarray,
        sim_time: float,
        step_count: int,
    ) -> np.ndarray:
        qj = data.qpos[7:]
        dqj = data.qvel[6:]
        quat = data.qpos[3:7]
        omega = data.qvel[3:6]

        qj = (qj - self.config.default_angles) * self.config.dof_pos_scale
        dqj = dqj * self.config.dof_vel_scale
        gravity_orientation = get_gravity_orientation(quat)
        omega = omega * self.config.ang_vel_scale

        period = 0.8
        phase = (sim_time % period) / period
        sin_phase = np.sin(2 * np.pi * phase)
        cos_phase = np.cos(2 * np.pi * phase)

        obs = self.obs_buffer.obs
        obs[:3] = omega
        obs[3:6] = gravity_orientation
        obs[6:9] = cmd * self.config.cmd_scale
        obs[9 : 9 + self.config.num_actions] = qj
        obs[9 + self.config.num_actions : 9 + 2 * self.config.num_actions] = dqj
        obs[9 + 2 * self.config.num_actions : 9 + 3 * self.config.num_actions] = action
        obs[9 + 3 * self.config.num_actions : 9 + 3 * self.config.num_actions + 2] = (
            np.array([sin_phase, cos_phase], dtype=np.float32)
        )
        for attack in self.obs_attacks:
            attack.apply(obs, self.context, sim_time, step_count)

        return obs.copy()


class G1MujocoRunner:
    def __init__(
        self,
        config: DeployConfig,
        policy: TorchScriptPolicy,
        cmd: Optional[np.ndarray] = None,
        attacks: Optional[Iterable[Attack]] = None,
        obs_attacks: Optional[list[ObservationAttack]] = None,
    ) -> None:
        self.config = config
        self.policy = policy
        self.cmd = (
            np.array(cmd, dtype=np.float32)
            if cmd is not None
            else self.config.cmd_init.copy()
        )
        self.attacks = list(attacks) if attacks is not None else []

        self.model = mujoco.MjModel.from_xml_path(str(self.config.xml_path))
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = self.config.simulation_dt
        self._ground_geom_ids = resolve_ground_geom_ids(
            self.model, self.config.ground_geom_names
        )

        self.obs_builder = G1ObservationBuilder(config, obs_attacks=obs_attacks)

        self.action = np.zeros(self.config.num_actions, dtype=np.float32)
        self.target_dof_pos = self.config.default_angles.copy()
        self.counter = 0
        self.sim_time = 0.0

    def reset(self) -> SimState:
        return self.reset_with_perturbation()

    def reset_with_perturbation(
        self,
        joint_pos_offset: Optional[np.ndarray] = None,
        joint_vel_offset: Optional[np.ndarray] = None,
    ) -> SimState:
        mujoco.mj_resetData(self.model, self.data)
        align_heightfield_baseline(self.model, self.data)

        num_joints = self.config.num_actions
        base_angles = self.config.default_angles.copy()

        if joint_pos_offset is not None:
            offset = np.array(joint_pos_offset, dtype=np.float32)
            if offset.shape[0] != num_joints:
                raise ValueError(
                    f"joint_pos_offset expects {num_joints} values, got {offset.shape[0]}"
                )
            base_angles = base_angles + offset
        self.data.qpos[7 : 7 + num_joints] = base_angles

        if joint_vel_offset is not None:
            vel_offset = np.array(joint_vel_offset, dtype=np.float32)
            if vel_offset.shape[0] != num_joints:
                raise ValueError(
                    f"joint_vel_offset expects {num_joints} values, got {vel_offset.shape[0]}"
                )
            self.data.qvel[6 : 6 + num_joints] = vel_offset
        else:
            self.data.qvel[6 : 6 + num_joints] = 0.0

        mujoco.mj_forward(self.model, self.data)

        self.action = np.zeros(self.config.num_actions, dtype=np.float32)
        self.target_dof_pos = base_angles.copy()
        self.counter = 0
        self.sim_time = 0.0
        self.obs_builder.reset()
        for attack in self.attacks:
            attack.reset(self.model, self.data)
        return self._snapshot_state()

    def step(self) -> SimState:
        tau = pd_control(
            self.target_dof_pos,
            self.data.qpos[7:],
            self.config.kps,
            np.zeros_like(self.config.kds),
            self.data.qvel[6:],
            self.config.kds,
        )
        self.data.ctrl[:] = tau
        for attack in self.attacks:
            attack.apply(self.model, self.data, self.sim_time, self.counter)
        mujoco.mj_step(self.model, self.data)

        self.counter += 1
        self.sim_time += self.config.simulation_dt

        if self.counter % self.config.control_decimation == 0:
            obs = self.obs_builder.build(
                self.data, self.action, self.cmd, self.sim_time, self.counter
            )
            self.action = self.policy.act(obs)
            self.target_dof_pos = self.action * self.config.action_scale + self.config.default_angles

        return self._snapshot_state()

    def apply_joint_perturbation(
        self,
        joint_pos_offset: Optional[np.ndarray],
        joint_vel_offset: Optional[np.ndarray],
    ) -> None:
        num_joints = self.config.num_actions
        if joint_pos_offset is not None:
            offset = np.array(joint_pos_offset, dtype=np.float32)
            if offset.shape[0] != num_joints:
                raise ValueError(
                    f"joint_pos_offset expects {num_joints} values, got {offset.shape[0]}"
                )
            self.data.qpos[7 : 7 + num_joints] += offset
        if joint_vel_offset is not None:
            offset = np.array(joint_vel_offset, dtype=np.float32)
            if offset.shape[0] != num_joints:
                raise ValueError(
                    f"joint_vel_offset expects {num_joints} values, got {offset.shape[0]}"
                )
            self.data.qvel[6 : 6 + num_joints] += offset
        mujoco.mj_forward(self.model, self.data)

    def _snapshot_state(self) -> SimState:
        stability_margin = None
        if self._ground_geom_ids:
            com_proj = self.data.subtree_com[0, :2].copy()
            support_poly = get_support_polygon(
                self.model, self.data, self._ground_geom_ids
            )
            stability_margin = calculate_stability_margin(com_proj, support_poly)
        return SimState(
            time=self.sim_time,
            qpos=self.data.qpos.copy(),
            qvel=self.data.qvel.copy(),
            action=self.action.copy(),
            cmd=self.cmd.copy(),
            stability_margin=stability_margin,
        )
