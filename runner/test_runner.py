from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import time

import mujoco.viewer

from robostl.core.config import DeployConfig
from robostl.metrics.basic_walking import WalkingMetrics
from robostl.specs.stl_evaluator import WalkingSTLSpec
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
            attacks=attacks,
            obs_attacks=obs_attacks,
        )

    def run_episode(self) -> EpisodeResult:
        max_steps = int(self.config.simulation_duration / self.config.simulation_dt)
        state = self.env.reset()
        self.metrics.reset(state)

        if self.render:
            with mujoco.viewer.launch_passive(self.env.model, self.env.data) as viewer:
                for _ in range(max_steps):
                    step_start = time.time()
                    state = self.env.step()
                    self.metrics.update(state)
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
                self.metrics.update(state)
                if self.stop_on_fall and self.metrics.fallen:
                    break

        metrics = self.metrics.finalize(state)
        height_threshold = max(
            self.metrics.min_height_abs,
            float(self.metrics.start_state.base_pos[2]) * self.metrics.min_height_ratio,
        )
        stl_spec = WalkingSTLSpec(
            height_threshold=height_threshold,
            max_tilt_deg=self.metrics.max_tilt_deg,
        )
        stl_result = stl_spec.evaluate(self.metrics.build_trace())
        metrics["stl_robustness"] = stl_result.robustness
        metrics["stl_height_robustness"] = stl_result.details.get("height_robustness")
        metrics["stl_tilt_robustness"] = stl_result.details.get("tilt_robustness")
        return EpisodeResult(
            metrics=metrics,
            stl={
                "ok": stl_result.ok,
                "robustness": stl_result.robustness,
                **stl_result.details,
            },
        )
