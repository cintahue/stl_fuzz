#!/usr/bin/env python3
"""
Layer 3: CROWN灵敏度引导的状态空间精细搜索

这是RoboSTL-Fuzz三层搜索架构的第三层。
与 run_layer3_verification.py (诊断分析) 不同，
本模块执行真正的搜索优化，在危险区域内寻找更极端的失效状态。

核心创新:
1. CROWN灵敏度引导 - 使用形式化验证技术识别高风险状态
2. CMA-ES局部精搜 - 在灵敏度引导下进行局部优化
3. 关节空间扰动 - 直接在29D状态空间搜索

方案参考:
- Layer 3应在Layer 2确定的危险相位内，对关节级状态进行精细搜索
- 使用CROWN作为"探雷器"，快速识别"高风险状态"
- 引导CMA-ES搜索方向，而非盲目采样
"""

from __future__ import annotations

if __package__ is None:
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Tuple, List

import numpy as np
import torch

from robostl.attacks.force import ForcePerturbation
from robostl.attacks.terrain import FloorFrictionModifier
from robostl.core.config import DeployConfig
from robostl.policies.torchscript import TorchScriptPolicy
from robostl.runner.test_runner import EpisodeResult, WalkingTestRunner
from robostl.search.cmaes import CMAES
from robostl.tasks.walking import WalkingTask


@dataclass
class SensitivityResult:
    """CROWN灵敏度分析结果"""
    sensitivity_score: float  # 整体灵敏度分数
    output_range: np.ndarray  # 各输出维度的范围
    sensitive_dims: np.ndarray  # 最敏感的输入维度索引
    lipschitz_estimate: float  # 局部Lipschitz常数估计


@dataclass
class Layer3SearchConfig:
    """Layer 3 搜索配置"""
    # 灵敏度预扫描
    prescan_samples: int = 50
    epsilon: float = 0.02
    
    # 局部搜索
    local_iterations: int = 30
    local_population: int = 16
    local_sigma: float = 0.1
    
    # 状态空间边界
    state_perturbation_scale: float = 0.05  # 关节位置扰动幅度
    vel_perturbation_scale: float = 0.02  # 关节速度扰动幅度
    joint_pos_scales: Optional[np.ndarray] = None  # 每个关节位置扰动幅度
    joint_vel_scales: Optional[np.ndarray] = None  # 每个关节速度扰动幅度
    sensitive_scale_factor: float = 0.5  # 敏感维度缩放
    nonsensitive_scale_factor: float = 1.0  # 非敏感维度缩放
    
    # 早停
    target_robustness: Optional[float] = None


@dataclass
class Layer3Result:
    """Layer 3 搜索结果"""
    case_rank: int
    original_robustness: float
    refined_robustness: float
    improvement: float
    best_state_perturbation: np.ndarray
    best_push_start: float
    sensitivity_analysis: dict
    search_trajectory: List[float] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Layer 3: CROWN-guided state space refinement search."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("layer2_results.json"),
        help="Layer 2 results JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("layer3_search_results.json"),
        help="Output JSON file.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DeployConfig.default_config_path(),
        help="Path to deploy config yaml.",
    )
    parser.add_argument(
        "--policy-path",
        type=Path,
        default=None,
        help="Override policy path.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="Number of top Layer 2 results to refine.",
    )
    parser.add_argument(
        "--prescan-samples",
        type=int,
        default=50,
        help="Number of states to prescan for sensitivity.",
    )
    parser.add_argument(
        "--local-iterations",
        type=int,
        default=30,
        help="CMA-ES iterations for local search.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.02,
        help="L-infinity perturbation for CROWN analysis.",
    )
    parser.add_argument(
        "--state-perturbation-scale",
        type=float,
        default=0.05,
        help="Max joint position perturbation magnitude.",
    )
    parser.add_argument(
        "--vel-perturbation-scale",
        type=float,
        default=0.02,
        help="Max joint velocity perturbation magnitude.",
    )
    parser.add_argument(
        "--joint-pos-scales",
        type=str,
        default=None,
        help="Comma-separated per-joint position scales (len=1 or num_joints).",
    )
    parser.add_argument(
        "--joint-vel-scales",
        type=str,
        default=None,
        help="Comma-separated per-joint velocity scales (len=1 or num_joints).",
    )
    parser.add_argument(
        "--sensitive-scale-factor",
        type=float,
        default=0.5,
        help="Scale factor for sensitive dimensions.",
    )
    parser.add_argument(
        "--nonsensitive-scale-factor",
        type=float,
        default=1.0,
        help="Scale factor for non-sensitive dimensions.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable rendering.",
    )
    parser.add_argument(
        "--use-crown",
        action="store_true",
        default=True,
        help="Use CROWN sensitivity guidance.",
    )
    parser.add_argument(
        "--no-crown",
        dest="use_crown",
        action="store_false",
        help="Disable CROWN (use uniform sampling).",
    )
    return parser.parse_args()


def _snapshot_memory(
    module: torch.jit.ScriptModule,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    if hasattr(module, "hidden_state") and hasattr(module, "cell_state"):
        hidden = module.hidden_state.detach().clone()
        cell = module.cell_state.detach().clone()
        return hidden, cell
    return None


def _restore_memory(
    module: torch.jit.ScriptModule,
    snapshot: Optional[Tuple[torch.Tensor, torch.Tensor]],
) -> None:
    if snapshot is None:
        return
    hidden, cell = snapshot
    if hasattr(module, "hidden_state"):
        module.hidden_state.copy_(hidden)
    if hasattr(module, "cell_state"):
        module.cell_state.copy_(cell)


def _safe_forward_np(
    module: torch.jit.ScriptModule,
    obs: np.ndarray,
    snapshot: Optional[Tuple[torch.Tensor, torch.Tensor]],
) -> np.ndarray:
    x = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)
    with torch.no_grad():
        _restore_memory(module, snapshot)
        y = module(x).squeeze(0).cpu().numpy()
        _restore_memory(module, snapshot)
    return y


def _finite_difference_jac(
    module: torch.jit.ScriptModule,
    obs: np.ndarray,
    epsilon: float,
) -> Tuple[np.ndarray, np.ndarray]:
    snapshot = _snapshot_memory(module)
    step = max(1e-4, float(epsilon) * 0.1)

    nominal = _safe_forward_np(module, obs, snapshot)
    action_dim = int(nominal.shape[0])
    obs_dim = int(obs.shape[0])
    jac = np.zeros((action_dim, obs_dim), dtype=np.float32)

    for i in range(obs_dim):
        obs_plus = obs.copy()
        obs_minus = obs.copy()
        obs_plus[i] += step
        obs_minus[i] -= step
        y_plus = _safe_forward_np(module, obs_plus, snapshot)
        y_minus = _safe_forward_np(module, obs_minus, snapshot)
        jac[:, i] = (y_plus - y_minus) / (2.0 * step)

    return jac, nominal


class CROWNSensitivityAnalyzer:
    """
    CROWN灵敏度分析器
    
    使用形式化验证技术计算策略网络的局部灵敏度，
    作为"探雷器"引导后续搜索。
    """
    
    def __init__(self, policy: TorchScriptPolicy, epsilon: float = 0.02, use_crown: bool = True):
        self.policy = policy
        self.epsilon = epsilon
        self._crown_available = use_crown and self._check_crown_availability()
    
    def _check_crown_availability(self) -> bool:
        """检查auto_LiRPA是否可用"""
        try:
            from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
            if hasattr(self.policy.module, "hidden_state"):
                print("[Warning] LSTM policy detected, disabling auto_LiRPA.")
                return False
            return True
        except ImportError:
            print("[Warning] auto_LiRPA not available, using linearized bounds.")
            return False
    
    def compute_sensitivity(self, obs: np.ndarray) -> SensitivityResult:
        """
        计算策略在给定观测下的灵敏度
        
        原理:
        - CROWN计算ε-球内输出的线性边界 [lb, ub]
        - 边界越宽 (ub - lb越大) → 网络越敏感 → 越危险
        
        Returns:
            SensitivityResult: 包含灵敏度分数、敏感维度等
        """
        obs_tensor = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)
        
        if self._crown_available:
            return self._compute_crown_sensitivity(obs_tensor)
        else:
            return self._compute_linearized_sensitivity(obs_tensor)
    
    def _compute_crown_sensitivity(self, obs_tensor: torch.Tensor) -> SensitivityResult:
        """使用auto_LiRPA计算CROWN边界"""
        from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
        
        try:
            # 创建bounded module
            bounded_model = BoundedModule(self.policy.module, obs_tensor)
            
            # 定义扰动
            ptb = PerturbationLpNorm(norm=np.inf, eps=self.epsilon)
            bounded_obs = BoundedTensor(obs_tensor, ptb)
            
            # 计算CROWN边界
            lb, ub = bounded_model.compute_bounds(x=(bounded_obs,), method="CROWN")
            
            lb = lb.squeeze(0).detach().cpu().numpy()
            ub = ub.squeeze(0).detach().cpu().numpy()
            output_range = ub - lb
            
            # 灵敏度分数 = 最大输出变化
            sensitivity_score = float(np.max(output_range))
            
            # Lipschitz估计 = 最大输出变化 / epsilon
            lipschitz_estimate = sensitivity_score / self.epsilon
            
            # 敏感维度 (通过梯度分析)
            sensitive_dims = self._compute_sensitive_dims(obs_tensor)
            
            return SensitivityResult(
                sensitivity_score=sensitivity_score,
                output_range=output_range,
                sensitive_dims=sensitive_dims,
                lipschitz_estimate=lipschitz_estimate,
            )
            
        except RuntimeError as e:
            # CROWN失败时回退到线性化方法
            print(f"[Warning] CROWN failed: {e}, using linearized fallback.")
            return self._compute_linearized_sensitivity(obs_tensor)
    
    def _compute_linearized_sensitivity(self, obs_tensor: torch.Tensor) -> SensitivityResult:
        """使用数值微分计算线性化灵敏度"""
        obs = obs_tensor.squeeze(0).numpy()
        jac, nominal = _finite_difference_jac(self.policy.module, obs, self.epsilon)
        obs_dim = obs.shape[0]

        delta = np.abs(jac).dot(np.full(obs_dim, self.epsilon, dtype=np.float32))
        output_range = 2 * delta
        
        # 灵敏度分数
        sensitivity_score = float(np.max(output_range))
        
        # Lipschitz估计
        lipschitz_estimate = float(np.max(np.linalg.norm(jac, axis=0)))
        
        # 敏感维度 (Jacobian列范数最大的维度)
        dim_sensitivity = np.linalg.norm(jac, axis=0)
        sensitive_dims = np.argsort(dim_sensitivity)[-10:][::-1]  # Top 10
        
        return SensitivityResult(
            sensitivity_score=sensitivity_score,
            output_range=output_range,
            sensitive_dims=sensitive_dims,
            lipschitz_estimate=lipschitz_estimate,
        )
    
    def _compute_sensitive_dims(self, obs_tensor: torch.Tensor) -> np.ndarray:
        """通过梯度分析识别敏感输入维度"""
        obs = obs_tensor.squeeze(0).numpy()
        jac, _ = _finite_difference_jac(self.policy.module, obs, self.epsilon)
        dim_sensitivity = np.linalg.norm(jac, axis=0)
        sensitive_dims = np.argsort(dim_sensitivity)[-10:][::-1]
        return sensitive_dims


class Layer3StateSearcher:
    """
    Layer 3 状态空间搜索器
    
    在Layer 2确定的危险相位内，对关节级状态进行精细搜索。
    使用CROWN灵敏度引导CMA-ES搜索方向。
    """
    
    def __init__(
        self,
        runner: WalkingTestRunner,
        analyzer: CROWNSensitivityAnalyzer,
        config: Layer3SearchConfig,
    ):
        self.runner = runner
        self.analyzer = analyzer
        self.config = config
        
        # 状态空间维度 (关节位置 + 关节速度)
        self.num_joints = self.runner.config.num_actions
        self.state_dim = self.num_joints * 2  # 关节位置 + 关节速度

        if config.joint_pos_scales is not None:
            if config.joint_pos_scales.shape[0] != self.num_joints:
                raise ValueError("joint_pos_scales length mismatch.")
        if config.joint_vel_scales is not None:
            if config.joint_vel_scales.shape[0] != self.num_joints:
                raise ValueError("joint_vel_scales length mismatch.")
    
    def search(
        self,
        base_push: Optional[np.ndarray],
        base_friction: Optional[np.ndarray],
        base_push_start: float,
        push_duration: float,
        push_body: str,
    ) -> Tuple[float, np.ndarray, List[SensitivityResult], Optional[EpisodeResult]]:
        """
        执行状态空间搜索
        
        Args:
            base_push: 基础推力参数
            base_friction: 基础摩擦参数
            base_push_start: 基础攻击时刻
            push_duration: 推力持续时间
            push_body: 推力作用点
        
        Returns:
            best_robustness: 找到的最低鲁棒性
            best_perturbation: 最佳状态扰动
            sensitivity_history: 灵敏度分析历史
        """
        # Phase 1: 灵敏度预扫描
        print("  [Phase 1] Sensitivity prescan...")
        sensitivity_samples = self._prescan_sensitivity(
            base_push, base_friction, base_push_start, push_duration, push_body
        )
        
        # Phase 2: 识别高风险状态
        print("  [Phase 2] Identifying high-risk states...")
        high_risk_states = sorted(
            sensitivity_samples, 
            key=lambda x: -x[1].sensitivity_score
        )[:10]  # Top 10
        
        # Phase 3: CMA-ES局部搜索
        print("  [Phase 3] CMA-ES local search...")
        best_robustness = float("inf")
        best_perturbation = np.zeros(self.state_dim)
        best_episode: Optional[EpisodeResult] = None
        search_trajectory = []
        
        for obs, sens in high_risk_states:
            robustness, perturbation, episode = self._local_search(
                obs, sens,
                base_push, base_friction, base_push_start,
                push_duration, push_body
            )
            search_trajectory.append(robustness)
            
            if robustness < best_robustness:
                best_robustness = robustness
                best_perturbation = perturbation
                best_episode = episode
                print(f"    New best: {robustness:.4f}")
        
        return best_robustness, best_perturbation, [s for _, s in sensitivity_samples], best_episode
    
    def _prescan_sensitivity(
        self,
        base_push: Optional[np.ndarray],
        base_friction: Optional[np.ndarray],
        base_push_start: float,
        push_duration: float,
        push_body: str,
    ) -> List[Tuple[np.ndarray, SensitivityResult]]:
        """
        预扫描多个状态的灵敏度
        
        在攻击时刻附近采样多个状态，评估其灵敏度。
        """
        samples = []
        
        # 设置攻击
        attacks = self._build_attacks(
            base_push, base_friction, base_push_start, push_duration, push_body
        )
        self.runner.env.attacks = attacks
        
        # 运行仿真并采样
        env = self.runner.env
        state = env.reset()
        dt = self.runner.config.simulation_dt
        
        # 在攻击时刻前后采样
        sample_window = 0.1  # 采样窗口 ±0.1s
        sample_start = max(0, base_push_start - sample_window)
        sample_end = base_push_start + push_duration + sample_window
        
        sample_interval = (sample_end - sample_start) / self.config.prescan_samples
        next_sample_time = sample_start
        
        max_steps = int(self.runner.config.simulation_duration / dt)
        for _ in range(max_steps):
            state = env.step()
            
            if state.time >= next_sample_time and state.time <= sample_end:
                # 构建观测
                obs = env.obs_builder.build(
                    env.data, env.action, env.cmd, env.sim_time, env.counter
                )
                
                # 计算灵敏度
                sens = self.analyzer.compute_sensitivity(obs)
                samples.append((obs.copy(), sens))
                
                next_sample_time += sample_interval
                
                if len(samples) >= self.config.prescan_samples:
                    break
            
            if state.time > sample_end:
                break
        
        return samples
    
    def _local_search(
        self,
        center_obs: np.ndarray,
        sensitivity: SensitivityResult,
        base_push: Optional[np.ndarray],
        base_friction: Optional[np.ndarray],
        base_push_start: float,
        push_duration: float,
        push_body: str,
    ) -> Tuple[float, np.ndarray, Optional[EpisodeResult]]:
        """
        在给定状态周围进行CMA-ES局部搜索
        
        使用灵敏度信息调整搜索分布。
        """
        # 初始化CMA-ES
        mean = np.zeros(self.state_dim, dtype=np.float32)
        
        # 根据灵敏度调整sigma
        # 在敏感维度上使用更小的sigma (精细搜索)
        sigma = self.config.local_sigma
        
        if self.config.joint_pos_scales is not None:
            pos_scales = self.config.joint_pos_scales.copy()
        else:
            pos_scales = np.ones(self.num_joints, dtype=np.float32) * self.config.state_perturbation_scale
        if self.config.joint_vel_scales is not None:
            vel_scales = self.config.joint_vel_scales.copy()
        else:
            vel_scales = np.ones(self.num_joints, dtype=np.float32) * self.config.vel_perturbation_scale

        pos_scales, vel_scales = self._apply_sensitivity_guidance(
            pos_scales, vel_scales, sensitivity
        )

        low = np.concatenate([-pos_scales, -vel_scales])
        high = np.concatenate([pos_scales, vel_scales])

        optimizer = CMAES(
            mean=mean,
            sigma=sigma,
            bounds=(low, high),
            population_size=self.config.local_population,
        )
        
        best_robustness = float("inf")
        best_perturbation = mean.copy()
        best_episode: Optional[EpisodeResult] = None
        
        for _ in range(self.config.local_iterations):
            candidates = optimizer.ask()
            losses = np.zeros(len(candidates), dtype=np.float32)
            
            for idx, perturbation in enumerate(candidates):
                # 评估扰动后的鲁棒性
                robustness, episode = self._evaluate_perturbation(
                    perturbation,
                    base_push, base_friction, base_push_start,
                    push_duration, push_body
                )
                losses[idx] = robustness
                
                if robustness < best_robustness:
                    best_robustness = robustness
                    best_perturbation = perturbation.copy()
                    best_episode = episode
            
            optimizer.tell(losses)
        
        return best_robustness, best_perturbation, best_episode
    
    def _evaluate_perturbation(
        self,
        perturbation: np.ndarray,
        base_push: Optional[np.ndarray],
        base_friction: Optional[np.ndarray],
        base_push_start: float,
        push_duration: float,
        push_body: str,
    ) -> Tuple[float, EpisodeResult]:
        """评估状态扰动的鲁棒性"""
        pos_offset = perturbation[: self.num_joints]
        vel_offset = perturbation[self.num_joints :]

        # 构建攻击
        attacks = self._build_attacks(
            base_push, base_friction, base_push_start, push_duration, push_body
        )
        self.runner.env.attacks = attacks
        
        # 运行仿真：在攻击时刻施加扰动
        result = self.runner.run_episode_with_midpoint_perturbation(
            perturbation_time=base_push_start,
            joint_pos_offset=pos_offset,
            joint_vel_offset=vel_offset,
        )

        return float(result.metrics.get("stl_robustness", 0.0)), result

    def _apply_sensitivity_guidance(
        self,
        pos_scales: np.ndarray,
        vel_scales: np.ndarray,
        sensitivity: SensitivityResult,
    ) -> Tuple[np.ndarray, np.ndarray]:
        pos_start = 9
        pos_end = pos_start + self.num_joints
        vel_start = pos_end
        vel_end = vel_start + self.num_joints

        pos_mask = np.zeros(self.num_joints, dtype=bool)
        vel_mask = np.zeros(self.num_joints, dtype=bool)

        for idx in sensitivity.sensitive_dims:
            if pos_start <= idx < pos_end:
                pos_mask[idx - pos_start] = True
            elif vel_start <= idx < vel_end:
                vel_mask[idx - vel_start] = True

        pos_scales *= self.config.nonsensitive_scale_factor
        vel_scales *= self.config.nonsensitive_scale_factor
        pos_scales[pos_mask] *= self.config.sensitive_scale_factor
        vel_scales[vel_mask] *= self.config.sensitive_scale_factor

        return pos_scales, vel_scales
    
    def _build_attacks(
        self,
        push: Optional[np.ndarray],
        friction: Optional[np.ndarray],
        push_start: float,
        push_duration: float,
        push_body: str,
    ) -> list:
        """构建攻击列表"""
        attacks = []
        if friction is not None:
            attacks.append(FloorFrictionModifier(friction=friction))
        if push is not None:
            attacks.append(
                ForcePerturbation(
                    body_name=push_body,
                    force=push,
                    start_time=push_start,
                    duration=push_duration,
                )
            )
        return attacks


def _extract_params(entry: dict) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[str]]:
    """从Layer 2结果中提取参数"""
    params = np.array(entry["params"], dtype=np.float32)
    names = entry.get("param_names", [])
    mapping = {name: float(value) for name, value in zip(names, params)}
    
    push = None
    if all(key in mapping for key in ("push_fx", "push_fy", "push_fz")):
        push = np.array(
            [mapping["push_fx"], mapping["push_fy"], mapping["push_fz"]],
            dtype=np.float32,
        )
    
    friction = None
    if all(key in mapping for key in ("fric_mu1", "fric_mu2", "fric_mu3")):
        friction = np.array(
            [mapping["fric_mu1"], mapping["fric_mu2"], mapping["fric_mu3"]],
            dtype=np.float32,
        )
    
    return push, friction, names


def main() -> None:
    args = parse_args()
    
    # 加载Layer 2结果
    layer2_data = json.loads(args.input.read_text(encoding="utf-8"))
    
    # 按最佳鲁棒性排序，取Top N
    sorted_entries = sorted(
        layer2_data,
        key=lambda e: e.get("best", {}).get("robustness", float("inf"))
    )
    top_entries = sorted_entries[:args.top_n]
    
    # 初始化
    config = DeployConfig.from_yaml(args.config)
    if args.policy_path is not None:
        config = DeployConfig(
            **{
                **config.__dict__,
                "policy_path": args.policy_path.expanduser().resolve(),
            }
        )
    
    policy = TorchScriptPolicy(config.policy_path)
    task = WalkingTask.from_config(config)
    runner = WalkingTestRunner(
        config=config,
        policy=policy,
        task=task,
        stop_on_fall=True,
        render=args.render,
        real_time=False,
        attacks=[],
        obs_attacks=None,
    )
    
    # 创建分析器和搜索器
    analyzer = CROWNSensitivityAnalyzer(
        policy, epsilon=args.epsilon, use_crown=args.use_crown
    )
    num_joints = config.num_actions

    def _parse_joint_scales(value: Optional[str], default_scale: float) -> np.ndarray:
        if value is None:
            return np.ones(num_joints, dtype=np.float32) * float(default_scale)
        parts = [p.strip() for p in value.split(",") if p.strip()]
        if len(parts) == 1:
            return np.ones(num_joints, dtype=np.float32) * float(parts[0])
        if len(parts) != num_joints:
            raise ValueError(
                f"Expected {num_joints} scales, got {len(parts)}: {value}"
            )
        return np.array([float(p) for p in parts], dtype=np.float32)

    joint_pos_scales = _parse_joint_scales(
        args.joint_pos_scales, args.state_perturbation_scale
    )
    joint_vel_scales = _parse_joint_scales(
        args.joint_vel_scales, args.vel_perturbation_scale
    )
    search_config = Layer3SearchConfig(
        prescan_samples=args.prescan_samples,
        epsilon=args.epsilon,
        local_iterations=args.local_iterations,
        state_perturbation_scale=args.state_perturbation_scale,
        vel_perturbation_scale=args.vel_perturbation_scale,
        joint_pos_scales=joint_pos_scales,
        joint_vel_scales=joint_vel_scales,
        sensitive_scale_factor=args.sensitive_scale_factor,
        nonsensitive_scale_factor=args.nonsensitive_scale_factor,
    )
    searcher = Layer3StateSearcher(runner, analyzer, search_config)
    
    # 对每个Top候选进行精细搜索
    results = []
    for idx, entry in enumerate(top_entries):
        rank = entry.get("rank", idx + 1)
        print(f"\n[{idx+1}/{len(top_entries)}] Refining Rank {rank}...")
        
        push, friction, names = _extract_params(entry)
        best = entry.get("best", {})
        push_start = best.get("push_start", 1.0)
        original_robustness = best.get("robustness", 0.0)
        
        # 执行搜索
        refined_robustness, best_perturbation, sensitivity_history, _best_episode = searcher.search(
            base_push=push,
            base_friction=friction,
            base_push_start=push_start,
            push_duration=0.2,
            push_body="pelvis",
        )
        
        improvement = original_robustness - refined_robustness
        print(f"  Original: {original_robustness:.4f} -> Refined: {refined_robustness:.4f}")
        print(f"  Improvement: {improvement:.4f}")
        
        results.append({
            "case_rank": rank,
            "param_names": names,
            "params": entry["params"],
            "original_robustness": original_robustness,
            "refined_robustness": refined_robustness,
            "improvement": improvement,
            "best_push_start": push_start,
            "best_perturbation": best_perturbation.tolist(),
            "sensitivity_summary": {
                "mean_score": float(np.mean([s.sensitivity_score for s in sensitivity_history])),
                "max_score": float(np.max([s.sensitivity_score for s in sensitivity_history])),
                "num_samples": len(sensitivity_history),
            },
        })
    
    # 保存结果
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nLayer 3 search results saved to {args.output}")
    
    # 打印总结
    print("\n" + "=" * 60)
    print("Layer 3 Search Summary")
    print("=" * 60)
    for r in sorted(results, key=lambda x: x["refined_robustness"]):
        print(f"  Rank {r['case_rank']}: {r['original_robustness']:.4f} -> {r['refined_robustness']:.4f} (Δ={r['improvement']:.4f})")


if __name__ == "__main__":
    main()
