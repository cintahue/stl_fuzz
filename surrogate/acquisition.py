"""Acquisition functions for surrogate-guided search.

Implements:
  - Expected Improvement (EI) for L1-level Bayesian optimization.
  - Diversity-aware selection for L2-L3 active learning.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

try:
    from scipy.stats import norm as _scipy_norm

    def _cdf(z: np.ndarray) -> np.ndarray:
        return _scipy_norm.cdf(z)

    def _pdf(z: np.ndarray) -> np.ndarray:
        return _scipy_norm.pdf(z)

except ImportError:
    # Fallback: crude normal CDF approximation
    def _cdf(z: np.ndarray) -> np.ndarray:  # type: ignore[misc]
        return 0.5 * (1.0 + np.tanh(z * 0.7978845608028654))

    def _pdf(z: np.ndarray) -> np.ndarray:  # type: ignore[misc]
        return np.exp(-0.5 * z ** 2) / 2.5066282746310002


def expected_improvement(
    mu: np.ndarray,
    sigma: np.ndarray,
    rho_best: float,
    xi: float = 0.01,
) -> np.ndarray:
    """Expected Improvement for *minimization*.

    EI is large when mu is small (below rho_best) or sigma is large.

    Args:
        mu:       predicted mean [N]
        sigma:    predicted std  [N]
        rho_best: current best (lowest) observed robustness
        xi:       exploration bonus

    Returns:
        EI values [N]
    """
    sigma = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-9)
    mu = np.asarray(mu, dtype=np.float64)
    improvement = rho_best - mu - xi
    z = improvement / sigma
    ei = improvement * _cdf(z) + sigma * _pdf(z)
    ei = np.where(sigma < 1e-9, 0.0, ei)
    return ei.astype(np.float32)


def select_by_acquisition(
    predictions: np.ndarray,
    existing_X: Optional[np.ndarray],
    n: int = 5,
    exploitation_ratio: float = 0.7,
) -> np.ndarray:
    """Select candidate indices balancing low-rho exploitation and diversity.

    Args:
        predictions:       predicted rho [N]
        existing_X:        already-sampled points [M, D] for diversity measure
        n:                 number of candidates to select
        exploitation_ratio: fraction from low-rho exploitation

    Returns:
        Selected indices [n]
    """
    N = len(predictions)
    if N == 0:
        return np.array([], dtype=int)

    n_exploit = max(1, int(round(n * exploitation_ratio)))
    n_explore = n - n_exploit

    sorted_asc = np.argsort(predictions)

    # Exploitation: lowest predicted rho
    exploit_idx = sorted_asc[:n_exploit]

    # Exploration: sample from indices not already in exploit set
    remaining = sorted_asc[n_exploit:]
    if n_explore > 0 and len(remaining) > 0:
        # Prefer candidates spread out in the ranking (not all near the top)
        stride = max(1, len(remaining) // (n_explore * 3 + 1))
        explore_pool = remaining[::stride][: n_explore * 3]
        if len(explore_pool) == 0:
            explore_pool = remaining
        n_take = min(n_explore, len(explore_pool))
        explore_idx = np.random.choice(explore_pool, size=n_take, replace=False)
    else:
        explore_idx = np.array([], dtype=int)

    selected = np.concatenate([exploit_idx, explore_idx]).astype(int)
    # Deduplicate while preserving order
    seen: set[int] = set()
    unique: list[int] = []
    for idx in selected:
        if idx not in seen:
            seen.add(idx)
            unique.append(int(idx))
    return np.array(unique[:n], dtype=int)
