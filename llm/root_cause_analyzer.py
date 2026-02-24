"""LLM-based failure root-cause analyzer.

Batches failure cases, calls the LLM to classify failure modes, and
summarises patterns to guide subsequent fuzzing (search-space bias + new seeds).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from robostl.llm.client import LLMClient

_SYSTEM_PROMPT_PATH = Path(__file__).parent / "prompts" / "root_cause_analysis.txt"


def _load_system_prompt() -> str:
    try:
        return _SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        return (
            "You are a robot safety analyst. Classify failure cases by mode "
            "(friction_slip, push_overload, terrain_trip, joint_error, combined) "
            "and return a JSON array."
        )


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class AnalysisResult:
    failure_id: str
    mode: str                        # one of the 5 failure modes
    critical_signal: str             # most-violated STL predicate
    contributing_factors: list[str]
    suggested_scenarios: list[str]   # natural-language prompts for SeedGenerator
    confidence: float                # 0–1


@dataclass
class PatternSummary:
    mode_distribution: dict[str, float]   # {"friction_slip": 0.4, ...}
    suggested_l1_bias: dict[str, float]   # {"friction_weight": 1.5, ...}
    new_scenario_descriptions: list[str]  # forwarded to SeedGenerator


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class FailureAnalyzer:
    """Batch failure analysis via LLM."""

    def __init__(
        self,
        client: LLMClient,
        batch_size: int = 5,
        dedup_threshold: float = 0.15,
    ) -> None:
        self.client = client
        self.batch_size = batch_size
        self.dedup_threshold = dedup_threshold
        self._system_prompt = _load_system_prompt()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_batch(self, failure_cases: list[dict]) -> list[AnalysisResult]:
        """Analyse a list of failure case dicts; returns AnalysisResult list."""
        results: list[AnalysisResult] = []
        for i in range(0, len(failure_cases), self.batch_size):
            batch = failure_cases[i : i + self.batch_size]
            batch_results = self._analyze_batch_chunk(batch)
            results.extend(batch_results)
        return results

    def should_analyze(
        self,
        new_case: dict,
        existing_analyses: list[AnalysisResult],
    ) -> bool:
        """Return True if new_case is sufficiently novel to warrant analysis."""
        if not existing_analyses:
            return True
        # Simple heuristic: always analyze (de-dup done at batch level)
        return True

    def summarize_patterns(
        self, analyses: list[AnalysisResult]
    ) -> PatternSummary:
        """Summarise failure patterns and derive search-bias suggestions."""
        if not analyses:
            return PatternSummary(
                mode_distribution={},
                suggested_l1_bias={},
                new_scenario_descriptions=[],
            )

        # Mode distribution
        counts: dict[str, int] = {}
        for a in analyses:
            counts[a.mode] = counts.get(a.mode, 0) + 1
        total = len(analyses)
        dist = {k: v / total for k, v in counts.items()}

        # Search-space bias hints
        bias: dict[str, float] = {}
        if dist.get("friction_slip", 0.0) > 0.4:
            bias["friction_weight"] = 1.5
        if dist.get("push_overload", 0.0) > 0.35:
            bias["push_weight"] = 1.3
        if dist.get("terrain_trip", 0.0) > 0.3:
            bias["terrain_weight"] = 1.5

        # Collect unique scenario suggestions
        seen: set[str] = set()
        scenarios: list[str] = []
        for a in analyses:
            for s in a.suggested_scenarios:
                if s and s not in seen:
                    seen.add(s)
                    scenarios.append(s)

        return PatternSummary(
            mode_distribution=dist,
            suggested_l1_bias=bias,
            new_scenario_descriptions=scenarios[:10],
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _analyze_batch_chunk(self, cases: list[dict]) -> list[AnalysisResult]:
        case_texts = []
        for case in cases:
            fid = str(case.get("iteration", "?"))
            case_texts.append(self._format_case(case, fid))

        user_content = "\n\n".join(case_texts)
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_content},
        ]

        results: list[AnalysisResult] = []
        try:
            raw_text = self.client.chat(messages, temperature=0.3)
            parsed = json.loads(raw_text)
            items = parsed if isinstance(parsed, list) else _unwrap_dict(parsed)
            for item in items:
                results.append(
                    AnalysisResult(
                        failure_id=str(item.get("failure_id", "")),
                        mode=str(item.get("mode", "combined")),
                        critical_signal=str(item.get("critical_signal", "unknown")),
                        contributing_factors=list(item.get("contributing_factors", [])),
                        suggested_scenarios=list(item.get("suggested_scenarios", [])),
                        confidence=float(item.get("confidence", 0.5)),
                    )
                )
        except Exception as exc:
            print(f"[FailureAnalyzer] Analysis error: {exc}")
        return results

    @staticmethod
    def _format_case(case: dict, case_id: str) -> str:
        push = case.get("push", [0, 0, 0])
        friction = case.get("friction", [1.0, 0.01, 0.001])
        terrain = case.get("terrain", {"mode": "flat"})
        phase = case.get("phase", 0.0)
        robustness = case.get("robustness", 0.0)

        stl_details = case.get("stl_details") or {}
        diag = stl_details.get("diagnostics", {}) if isinstance(stl_details, dict) else {}
        most_violated = diag.get("most_violated_predicate", "unknown")
        first_viol_time = diag.get("first_violation_time", "N/A")

        pos_strs: list[str] = []
        vel_strs: list[str] = []
        jp = case.get("joint_pos_offset")
        jv = case.get("joint_vel_offset")
        if jp is not None:
            arr = np.asarray(jp)
            for idx in np.where(np.abs(arr) > 1e-4)[0][:5]:
                pos_strs.append(f"j{idx}={arr[idx]:.3f}")
        if jv is not None:
            arr = np.asarray(jv)
            for idx in np.where(np.abs(arr) > 1e-4)[0][:5]:
                vel_strs.append(f"j{idx}={arr[idx]:.3f}")

        t_mode = terrain.get("mode", "flat")
        t_str = t_mode
        if t_mode == "pit":
            t_str += f", depth={terrain.get('depth', 0):.3f}m"
        elif t_mode == "bump":
            t_str += f", height={terrain.get('height', 0):.3f}m"

        return "\n".join(
            [
                f"案例 #{case_id}:",
                f"  推力: {push} N",
                f"  摩擦系数: {friction}",
                f"  地形: {t_str}",
                f"  攻击相位: {phase:.3f}（步态周期 {phase*100:.0f}%）",
                f"  STL鲁棒性: {robustness:.4f}",
                f"  最先违反谓词: {most_violated}（t={first_viol_time}s）",
                f"  关节位置偏移: {pos_strs or '无'}",
                f"  关节速度偏移: {vel_strs or '无'}",
            ]
        )


def _unwrap_dict(parsed: dict) -> list[dict]:
    """Try to extract a list from a JSON object response."""
    for v in parsed.values():
        if isinstance(v, list):
            return v
    return []
