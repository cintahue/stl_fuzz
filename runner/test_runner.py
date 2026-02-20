from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import time

import mujoco.viewer

from robostl.core.config import DeployConfig
from robostl.metrics.basic_walking import WalkingMetrics
from robostl.specs.spec_config import SpecConfig
from robostl.specs.walking_specs import CompositeWalkingSpec
from robostl.sim.g1_env import G1MujocoRunner
from robostl.policies.torchscript import TorchScriptPolicy
from robostl.tasks.walking import WalkingTask


@dataclass
class EpisodeResult:
    metrics: dict
    stl: dict


class WalkingTestRunner:
    def __init__(
        self,
        config: DeployConfig,
        policy: TorchScriptPolicy,
        task: WalkingTask,
        metrics: Optional[WalkingMetrics] = None,
        stop_on_fall: bool = True,
        render: bool = True,
        real_time: bool = True,
        attacks: Optional[list] = None,
        terrain: Optional[dict] = None,
        obs_attacks: Optional[list] = None,
    ) -> None:
        self.config = config
        self.policy = policy
        self.task = task
        self.metrics = metrics or WalkingMetrics()
        self.stop_on_fall = stop_on_fall
        self.render = render
        self.real_time = real_time

        self.env = G1MujocoRunner(
            config=config,
            policy=policy,
            cmd=task.command,
            terrain=terrain,
            attacks=attacks,
            obs_attacks=obs_attacks,
        )

    def run_episode(
        self,
        joint_pos_offset: Optional[np.ndarray] = None,
        joint_vel_offset: Optional[np.ndarray] = None,
    ) -> EpisodeResult:
        max_steps = int(self.config.simulation_duration / self.config.simulation_dt)
        if joint_pos_offset is not None or joint_vel_offset is not None:
            state = self.env.reset_with_perturbation(joint_pos_offset, joint_vel_offset)
        else:
            state = self.env.reset()
        self.metrics.reset(state)
        prev_action = self.env.action.copy()

        if self.render:
            with mujoco.viewer.launch_passive(self.env.model, self.env.data) as viewer:
                for _ in range(max_steps):
                    step_start = time.time()
                    state = self.env.step()
                    self.metrics.update(
                        state,
                        self.env.model,
                        self.env.data,
                        self.env.data.ctrl,
                        self.env.cmd,
                        prev_action,
                        self.env._ground_geom_ids,
                    )
                    prev_action = state.action.copy()
                    viewer.sync()
                    if self.stop_on_fall and self.metrics.fallen:
                        break
                    if self.real_time:
                        time_until_next = self.config.simulation_dt - (time.time() - step_start)
                        if time_until_next > 0:
                            time.sleep(time_until_next)
                    if not viewer.is_running():
                        break
        else:
            for _ in range(max_steps):
                state = self.env.step()
                self.metrics.update(
                    state,
                    self.env.model,
                    self.env.data,
                    self.env.data.ctrl,
                    self.env.cmd,
                    prev_action,
                    self.env._ground_geom_ids,
                )
                prev_action = state.action.copy()
                if self.stop_on_fall and self.metrics.fallen:
                    break

        metrics = self.metrics.finalize(state)
        spec_config = SpecConfig.from_dict(getattr(self.config, "stl_config", None))
        height_threshold = max(
            spec_config.h_min,
            float(self.metrics.start_state.base_pos[2]) * self.metrics.min_height_ratio,
        )
        spec_config.h_min = height_threshold
        stl_spec = CompositeWalkingSpec(spec_config)
        stl_result = stl_spec.evaluate(self.metrics.build_trace())
        metrics["stl_robustness"] = stl_result.robustness
        metrics["stl_safety_robustness"] = stl_result.safety.robustness
        metrics["stl_stability_robustness"] = stl_result.stability.robustness
        metrics["stl_performance_robustness"] = (
            stl_result.performance.robustness if stl_result.performance else None
        )
        metrics["stl_first_violation_time"] = stl_result.diagnostics.first_violation_time
        metrics["stl_most_violated_predicate"] = stl_result.diagnostics.most_violated_predicate
        metrics["stl_details"] = stl_result.to_dict()
        metrics["stl_height_robustness"] = (
            stl_result.safety.details.get("height", {}).get("robustness")
            if stl_result.safety
            else None
        )
        metrics["stl_tilt_robustness"] = (
            stl_result.safety.details.get("tilt", {}).get("robustness")
            if stl_result.safety
            else None
        )
        return EpisodeResult(
            metrics=metrics,
            stl=stl_result.to_dict(),
        )

    def run_episode_with_midpoint_perturbation(
        self,
        perturbation_time: float,
        joint_pos_offset: Optional[np.ndarray] = None,
        joint_vel_offset: Optional[np.ndarray] = None,
    ) -> EpisodeResult:
        max_steps = int(self.config.simulation_duration / self.config.simulation_dt)
        state = self.env.reset()
        self.metrics.reset(state)
        prev_action = self.env.action.copy()

        applied = False

        if self.render:
            with mujoco.viewer.launch_passive(self.env.model, self.env.data) as viewer:
                for _ in range(max_steps):
                    step_start = time.time()
                    state = self.env.step()
                    if not applied and state.time >= perturbation_time:
                        self.env.apply_joint_perturbation(joint_pos_offset, joint_vel_offset)
                        applied = True
                    self.metrics.update(
                        state,
                        self.env.model,
                        self.env.data,
                        self.env.data.ctrl,
                        self.env.cmd,
                        prev_action,
                        self.env._ground_geom_ids,
                    )
                    prev_action = state.action.copy()
                    viewer.sync()
                    if self.stop_on_fall and self.metrics.fallen:
                        break
                    if self.real_time:
                        time_until_next = self.config.simulation_dt - (time.time() - step_start)
                        if time_until_next > 0:
                            time.sleep(time_until_next)
                    if not viewer.is_running():
                        break
        else:
            for _ in range(max_steps):
                state = self.env.step()
                if not applied and state.time >= perturbation_time:
                    self.env.apply_joint_perturbation(joint_pos_offset, joint_vel_offset)
                    applied = True
                self.metrics.update(
                    state,
                    self.env.model,
                    self.env.data,
                    self.env.data.ctrl,
                    self.env.cmd,
                    prev_action,
                    self.env._ground_geom_ids,
                )
                prev_action = state.action.copy()
                if self.stop_on_fall and self.metrics.fallen:
                    break

        metrics = self.metrics.finalize(state)
        spec_config = SpecConfig.from_dict(getattr(self.config, "stl_config", None))
        height_threshold = max(
            spec_config.h_min,
            float(self.metrics.start_state.base_pos[2]) * self.metrics.min_height_ratio,
        )
        spec_config.h_min = height_threshold
        stl_spec = CompositeWalkingSpec(spec_config)
        stl_result = stl_spec.evaluate(self.metrics.build_trace())
        metrics["stl_robustness"] = stl_result.robustness
        metrics["stl_safety_robustness"] = stl_result.safety.robustness
        metrics["stl_stability_robustness"] = stl_result.stability.robustness
        metrics["stl_performance_robustness"] = (
            stl_result.performance.robustness if stl_result.performance else None
        )
        metrics["stl_first_violation_time"] = stl_result.diagnostics.first_violation_time
        metrics["stl_most_violated_predicate"] = stl_result.diagnostics.most_violated_predicate
        metrics["stl_details"] = stl_result.to_dict()
        metrics["stl_height_robustness"] = (
            stl_result.safety.details.get("height", {}).get("robustness")
            if stl_result.safety
            else None
        )
        metrics["stl_tilt_robustness"] = (
            stl_result.safety.details.get("tilt", {}).get("robustness")
            if stl_result.safety
            else None
        )
        return EpisodeResult(
            metrics=metrics,
            stl=stl_result.to_dict(),
        )
