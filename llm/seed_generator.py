"""LLM-based scenario seed generator.

Converts natural-language scenario descriptions into L1 fuzzing parameter
dicts via an OpenAI-compatible LLM.  Also supports augmentation (generating
variants of existing seeds).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from robostl.llm.client import LLMClient

_SYSTEM_PROMPT_PATH = Path(__file__).parent / "prompts" / "scenario_to_params.txt"


def _load_system_prompt() -> str:
    try:
        return _SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        return (
            "You are a robot safety testing engineer. "
            "Convert scenario descriptions to simulation test parameters in JSON."
        )


class ScenarioSeedGenerator:
    """Generate fuzzing seeds from natural-language descriptions using an LLM."""

    # Hard bounds for clipping LLM output
    _PUSH_LO, _PUSH_HI = -80.0, 80.0
    _FRIC_BOUNDS = [(0.2, 2.0), (0.001, 0.02), (0.00001, 0.01)]
    _CMD_BOUNDS = [(-1.0, 1.0), (-0.3, 0.3), (-0.5, 0.5)]
    _CX_BOUNDS = (0.5, 2.0)
    _CY_BOUNDS = (-0.5, 0.5)
    _R_BOUNDS = (0.08, 0.20)
    _DH_BOUNDS = (0.01, 0.05)

    def __init__(
        self,
        client: LLMClient,
        batch_size: int = 5,
    ) -> None:
        self.client = client
        self.batch_size = batch_size
        self._system_prompt = _load_system_prompt()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_from_descriptions(self, descriptions: list[str]) -> list[dict]:
        """Generate parameter dicts from a list of natural-language descriptions."""
        if not descriptions:
            return []
        results: list[dict] = []
        for i in range(0, len(descriptions), self.batch_size):
            batch = descriptions[i : i + self.batch_size]
            results.extend(self._call_llm_batch(batch))
        return results

    def generate_augmented(
        self,
        base_seeds: list[dict],
        n_variants: int = 5,
    ) -> list[dict]:
        """Ask the LLM to generate variants inspired by existing seeds."""
        if not base_seeds:
            return []
        seed_summary = "\n".join(
            f"{j+1}. push={s.get('push',[0,0,0])}, "
            f"friction={s.get('friction',[1,0.01,0.001])}, "
            f"terrain={s.get('terrain',{}).get('mode','flat')}, "
            f"cmd={s.get('cmd',[0.5,0,0])}"
            for j, s in enumerate(base_seeds[:10])
        )
        user_msg = (
            f"以下是已知的危险测试参数组合：\n{seed_summary}\n\n"
            f"请基于这些参数的规律，推理出 {n_variants} 个类似但不同的场景，"
            "以JSON数组格式返回（每个元素符合 schema）。"
        )
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_msg},
        ]
        results: list[dict] = []
        try:
            raw_text = self.client.chat(messages, temperature=0.7)
            parsed = json.loads(raw_text)
            items = parsed if isinstance(parsed, list) else [parsed]
            for item in items:
                clipped = self._clip(item)
                if clipped:
                    results.append(clipped)
        except Exception as exc:
            print(f"[SeedGenerator] Augmentation error: {exc}")
        return results

    @staticmethod
    def save_seeds(seeds: list[dict], output_path: Path) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(seeds, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[SeedGenerator] Saved {len(seeds)} seeds to {output_path}")

    @staticmethod
    def load_seeds(input_path: Path) -> list[dict]:
        input_path = Path(input_path)
        if not input_path.exists():
            return []
        try:
            data = json.loads(input_path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except Exception as exc:
            print(f"[SeedGenerator] Load error: {exc}")
            return []

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _call_llm_batch(self, descriptions: list[str]) -> list[dict]:
        if len(descriptions) == 1:
            user_msg = (
                f"场景描述：{descriptions[0]}\n\n"
                "请根据上述场景，生成最可能导致机器人失稳的测试参数。"
            )
            messages = [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": user_msg},
            ]
            raw = self.client.chat_json(messages, temperature=0.7)
            if raw:
                clipped = self._clip(raw)
                return [clipped] if clipped else []
            return []

        # Multiple descriptions — ask for JSON array
        numbered = "\n".join(f"{j+1}. {d}" for j, d in enumerate(descriptions))
        user_msg = (
            "以下是多个场景描述，请为每个场景分别生成测试参数，"
            f"以JSON数组格式返回（共 {len(descriptions)} 个元素）：\n\n{numbered}"
        )
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_msg},
        ]
        results: list[dict] = []
        try:
            raw_text = self.client.chat(messages, temperature=0.7)
            parsed = json.loads(raw_text)
            items = parsed if isinstance(parsed, list) else [parsed]
            for item in items:
                clipped = self._clip(item)
                if clipped:
                    results.append(clipped)
        except Exception as exc:
            print(f"[SeedGenerator] Batch call error: {exc}")
        return results

    def _clip(self, params: dict) -> Optional[dict]:
        """Clip/validate LLM output to legal parameter ranges."""
        try:
            push_raw = params.get("push", [0.0, 0.0, 0.0])
            push = list(
                np.clip(np.asarray(push_raw, dtype=float), self._PUSH_LO, self._PUSH_HI)
            )

            fric_raw = params.get("friction", [1.0, 0.01, 0.001])
            friction = [
                float(np.clip(fric_raw[i], *self._FRIC_BOUNDS[i])) for i in range(3)
            ]

            cmd_raw = params.get("cmd", [0.5, 0.0, 0.0])
            cmd = [float(np.clip(cmd_raw[i], *self._CMD_BOUNDS[i])) for i in range(3)]

            terrain_raw = params.get("terrain", {"mode": "flat"})
            mode = terrain_raw.get("mode", "flat")
            if mode in ("pit", "bump"):
                cr = terrain_raw.get("center", [1.0, 0.0])
                center = [
                    float(np.clip(cr[0], *self._CX_BOUNDS)),
                    float(np.clip(cr[1], *self._CY_BOUNDS)),
                ]
                radius = float(np.clip(terrain_raw.get("radius", 0.12), *self._R_BOUNDS))
                dh = float(
                    np.clip(terrain_raw.get("depth_or_height", 0.03), *self._DH_BOUNDS)
                )
                terrain = {
                    "mode": mode,
                    "center": center,
                    "radius": radius,
                    ("depth" if mode == "pit" else "height"): dh,
                }
            else:
                terrain = {"mode": "flat"}

            return {
                "push": push,
                "friction": friction,
                "terrain": terrain,
                "cmd": cmd,
                "severity": int(np.clip(params.get("severity", 3), 1, 5)),
                "reasoning": str(params.get("reasoning", "")),
            }
        except Exception as exc:
            print(f"[SeedGenerator] Clip error: {exc}")
            return None
