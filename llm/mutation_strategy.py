"""LLM-based mutation strategy generator.

LLM 不直接生成参数向量，而是输出结构化的"变异策略"：
- 各参数维度的变异步长倍率（push_scale、friction_scale、cmd_scale）
- 变异均值偏移（push_bias、friction_bias、cmd_bias）
- 地形模式采样权重（terrain_mode_weights）

传统变异算子仍每轮执行，但其分布受 LLM 策略调控。
始终保留 random_exploration_ratio 比例的纯随机变异，
确保即使 LLM 策略偏差也不丧失探索能力。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from robostl.llm.client import LLMClient

_SYSTEM_PROMPT_PATH = Path(__file__).parent / "prompts" / "mutation_strategy.txt"


def _load_system_prompt() -> str:
    try:
        return _SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        return (
            "You are a fuzzing strategy coach for robot safety testing. "
            "Analyze failure cases and suggest mutation strategy adjustments in JSON."
        )


@dataclass
class MutationStrategy:
    """LLM 输出的变异策略数据结构。

    scale 字段 > 1 表示加大该维度的探索步长，< 1 表示缩小。
    bias 字段在 gaussian 变异的均值上叠加偏移量。
    """

    # 各参数维度变异步长倍率（乘以 sigma_base）
    push_scale: np.ndarray = field(
        default_factory=lambda: np.ones(3, dtype=np.float32)
    )
    friction_scale: np.ndarray = field(
        default_factory=lambda: np.ones(3, dtype=np.float32)
    )
    cmd_scale: np.ndarray = field(
        default_factory=lambda: np.ones(3, dtype=np.float32)
    )

    # 变异均值偏移（叠加在 base seed 参数上）
    push_bias: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )
    friction_bias: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )
    cmd_bias: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )

    # 地形模式采样权重（归一化后使用）
    terrain_mode_weights: dict = field(
        default_factory=lambda: {"flat": 0.33, "pit": 0.33, "bump": 0.34}
    )

    # 元信息
    confidence: float = 0.5
    valid_for_iterations: int = 25
    reasoning: str = ""


class MutationStrategyGenerator:
    """从失效案例生成变异策略（通过 LLM 调用）。"""

    def __init__(self, client: LLMClient) -> None:
        self.client = client
        self._system_prompt = _load_system_prompt()

    def generate_strategy(
        self,
        failure_cases: list[dict],
        pool_stats: Optional[dict] = None,
        valid_for_iterations: int = 25,
    ) -> MutationStrategy:
        """分析失效案例，生成变异策略。

        Args:
            failure_cases: 近期失效案例列表（raw dict 格式）
            pool_stats:    种子池分布统计（可选，辅助 LLM 判断探索盲区）
            valid_for_iterations: 策略默认有效迭代数

        Returns:
            MutationStrategy（LLM 调用失败时返回默认值）
        """
        if not failure_cases:
            return MutationStrategy()

        # 格式化失效案例摘要
        case_texts: list[str] = []
        for i, case in enumerate(failure_cases[-10:]):  # 最多 10 个
            push = case.get("push", [0, 0, 0])
            friction = case.get("friction", [1.0, 0.01, 0.001])
            terrain = case.get("terrain", {})
            cmd = case.get("cmd", [0.5, 0, 0])
            robustness = case.get("robustness", 0.0)
            stl_details = case.get("stl_details") or {}
            diag = (
                stl_details.get("diagnostics", {})
                if isinstance(stl_details, dict)
                else {}
            )
            most_violated = diag.get("most_violated_predicate", "unknown")

            case_texts.append(
                f"案例 #{i + 1}: push={push}, "
                f"friction={friction}, "
                f"terrain={terrain.get('mode', 'flat')}, "
                f"cmd={cmd}, "
                f"robustness={robustness:.3f}, "
                f"最先违反谓词={most_violated}"
            )

        pool_text = ""
        if pool_stats:
            pool_text = f"\n种子池统计：{json.dumps(pool_stats, ensure_ascii=False)}"

        user_content = (
            f"近期 {len(case_texts)} 个失效案例：\n"
            + "\n".join(case_texts)
            + pool_text
            + "\n\n请分析上述失效案例的规律，给出变异策略建议。"
        )

        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_content},
        ]

        try:
            raw = self.client.chat_json(messages, temperature=0.4)
            if raw:
                return self._parse_strategy(raw, valid_for_iterations)
        except Exception as exc:
            print(f"[MutationStrategy] Generation error: {exc}")

        return MutationStrategy()

    def _parse_strategy(self, data: dict, valid_for_iterations: int) -> MutationStrategy:
        """将 LLM JSON 响应解析为 MutationStrategy。"""

        def _scale_arr(key: str, default: list) -> np.ndarray:
            raw = data.get(key, default)
            # 限制倍率范围，防止 LLM 输出极端值
            return np.clip(np.array(raw, dtype=np.float32), 0.1, 5.0)

        def _bias_arr(key: str, default: list, max_abs: float) -> np.ndarray:
            raw = data.get(key, default)
            return np.clip(np.array(raw, dtype=np.float32), -max_abs, max_abs)

        terrain_weights = dict(data.get("terrain_mode_weights", {}))
        if not terrain_weights:
            terrain_weights = {"flat": 0.33, "pit": 0.33, "bump": 0.34}
        # 归一化
        total = sum(terrain_weights.values())
        if total > 0:
            terrain_weights = {k: float(v) / total for k, v in terrain_weights.items()}

        return MutationStrategy(
            push_scale=_scale_arr("push_scale", [1.0, 1.0, 1.0]),
            friction_scale=_scale_arr("friction_scale", [1.0, 1.0, 1.0]),
            cmd_scale=_scale_arr("cmd_scale", [1.0, 1.0, 1.0]),
            push_bias=_bias_arr("push_bias", [0.0, 0.0, 0.0], max_abs=30.0),
            friction_bias=_bias_arr("friction_bias", [0.0, 0.0, 0.0], max_abs=0.5),
            cmd_bias=_bias_arr("cmd_bias", [0.0, 0.0, 0.0], max_abs=0.3),
            terrain_mode_weights=terrain_weights,
            confidence=float(np.clip(data.get("confidence", 0.5), 0.0, 1.0)),
            valid_for_iterations=int(
                data.get("valid_for_iterations", valid_for_iterations)
            ),
            reasoning=str(data.get("reasoning", "")),
        )


def compute_pool_stats(seed_pool: list) -> dict:
    """计算种子池的分布统计，供 LLM 参考。"""
    if not seed_pool:
        return {}

    pushes = np.array([s.push for s in seed_pool], dtype=np.float32)
    frictions = np.array([s.friction for s in seed_pool], dtype=np.float32)
    cmds = np.array([s.cmd for s in seed_pool], dtype=np.float32)
    terrains = [s.terrain.get("mode", "flat") if s.terrain else "flat" for s in seed_pool]
    terrain_counts = {}
    for t in terrains:
        terrain_counts[t] = terrain_counts.get(t, 0) + 1

    return {
        "n_seeds": len(seed_pool),
        "push_mean": pushes.mean(axis=0).tolist(),
        "push_std": pushes.std(axis=0).tolist(),
        "friction_mean": frictions.mean(axis=0).tolist(),
        "friction_std": frictions.std(axis=0).tolist(),
        "cmd_mean": cmds.mean(axis=0).tolist(),
        "cmd_std": cmds.std(axis=0).tolist(),
        "terrain_dist": terrain_counts,
    }
