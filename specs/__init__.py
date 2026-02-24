"""STL 规约模块对外导出入口。

统一暴露常用数据结构与评估器，避免调用方直接依赖子模块路径。
"""

from robostl.specs.extended_result import Diagnostics, ExtendedSTLResult, LayerResult
from robostl.specs.spec_config import SpecConfig
from robostl.specs.walking_specs import CompositeWalkingSpec

__all__ = [
    "Diagnostics",
    "ExtendedSTLResult",
    "LayerResult",
    "SpecConfig",
    "CompositeWalkingSpec",
]
