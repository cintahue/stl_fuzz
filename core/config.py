from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from robostl.core.paths import get_robostl_root


@dataclass
class DeployConfig:
    policy_path: Path
    xml_path: Path
    simulation_duration: float
    simulation_dt: float
    control_decimation: int
    kps: np.ndarray
    kds: np.ndarray
    default_angles: np.ndarray
    ang_vel_scale: float
    dof_pos_scale: float
    dof_vel_scale: float
    action_scale: float
    cmd_scale: np.ndarray
    num_actions: int
    num_obs: int
    cmd_init: np.ndarray
    ground_geom_names: list[str]
    stl_config: dict = field(default_factory=dict)
    deviation_stl_config: dict = field(default_factory=dict)
    llm_config: dict = field(default_factory=dict)

    @staticmethod
    def default_config_path() -> Path:
        robostl_root = get_robostl_root()
        return robostl_root / "configs" / "g1_12dof_mujoco.yaml"

    @classmethod
    def from_yaml(cls, path: Path, robostl_root: Path | None = None) -> "DeployConfig":
        if robostl_root is None:
            robostl_root = get_robostl_root()

        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        def _resolve_path(value: Any) -> Path:
            if not isinstance(value, str):
                raise TypeError(f"Expected path string, got {type(value)}")
            resolved = value.replace("{ROBOSTL_ROOT}", str(robostl_root))
            resolved = resolved.replace("{LEGGED_GYM_ROOT_DIR}", str(robostl_root))
            return Path(resolved).expanduser().resolve()

        return cls(
            policy_path=_resolve_path(raw["policy_path"]),
            xml_path=_resolve_path(raw["xml_path"]),
            simulation_duration=float(raw["simulation_duration"]),
            simulation_dt=float(raw["simulation_dt"]),
            control_decimation=int(raw["control_decimation"]),
            kps=np.array(raw["kps"], dtype=np.float32),
            kds=np.array(raw["kds"], dtype=np.float32),
            default_angles=np.array(raw["default_angles"], dtype=np.float32),
            ang_vel_scale=float(raw["ang_vel_scale"]),
            dof_pos_scale=float(raw["dof_pos_scale"]),
            dof_vel_scale=float(raw["dof_vel_scale"]),
            action_scale=float(raw["action_scale"]),
            cmd_scale=np.array(raw["cmd_scale"], dtype=np.float32),
            num_actions=int(raw["num_actions"]),
            num_obs=int(raw["num_obs"]),
            cmd_init=np.array(raw["cmd_init"], dtype=np.float32),
            ground_geom_names=list(raw.get("ground_geom_names", ["terrain", "floor"])),
            stl_config=dict(raw.get("stl", {})),
            deviation_stl_config=dict(raw.get("deviation_stl", {})),
            llm_config=dict(raw.get("llm", {})),
        )
