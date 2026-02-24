"""Bayesian Optimizer for L1-level candidate suggestion.

Uses GPR + Expected Improvement with Sobol quasi-random candidate generation.
Does NOT maintain its own data; relies on DataBuffer and GlobalSurrogate.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

try:
    from scipy.stats.qmc import Sobol as _Sobol

    def _sobol_sample(n: int, d: int) -> np.ndarray:
        """Generate n Sobol samples in [0,1]^d."""
        sampler = _Sobol(d=d, scramble=True)
        # Sobol requires n to be a power of 2; round up
        m = int(np.ceil(np.log2(max(n, 2))))
        raw = sampler.random_base2(m)  # shape [2^m, d]
        return raw[:n]

    SOBOL_AVAILABLE = True
except ImportError:
    SOBOL_AVAILABLE = False

    def _sobol_sample(n: int, d: int) -> np.ndarray:  # type: ignore[misc]
        return np.random.uniform(0.0, 1.0, (n, d))


from robostl.surrogate.acquisition import expected_improvement


class BayesianOptimizer:
    """L1-level Bayesian optimizer: Sobol candidates → GPR + EI → suggest.

    Wraps GlobalSurrogate's GPR for Expected Improvement–based candidate
    selection over the L1 parameter space.
    """

    def __init__(
        self,
        bounds: np.ndarray,
        n_candidates: int = 1000,
        xi: float = 0.01,
    ) -> None:
        """
        Args:
            bounds:       shape [D, 2], columns = [lower, upper]
            n_candidates: Sobol candidates evaluated per call to suggest()
            xi:           EI exploration bonus
        """
        self.bounds = np.asarray(bounds, dtype=np.float64)
        self.n_candidates = n_candidates
        self.xi = xi
        self.dim = len(bounds)
        self._rho_best: float = float("inf")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def suggest(
        self,
        surrogate,
        rho_best: Optional[float] = None,
        n_suggestions: int = 1,
    ) -> List[np.ndarray]:
        """Return n_suggestions L1 candidates maximising Expected Improvement.

        Args:
            surrogate:     GlobalSurrogate with a fitted GPR
            rho_best:      current best (lowest) rho; uses internal tracker if None
            n_suggestions: how many candidates to return

        Returns:
            List of candidate L1 vectors (float32 ndarray, shape [D])
        """
        if rho_best is None:
            rho_best = self._rho_best

        candidates = self._scaled_sobol(self.n_candidates)

        # GPR prediction for all candidates
        mu_list, sigma_list = [], []
        for c in candidates:
            mu, sigma = surrogate.gpr_predict(c)
            mu_list.append(mu)
            sigma_list.append(sigma)

        mu_arr = np.array(mu_list, dtype=np.float64)
        sigma_arr = np.array(sigma_list, dtype=np.float64)

        ei_vals = expected_improvement(mu_arr, sigma_arr, rho_best, xi=self.xi)

        top_idx = np.argsort(ei_vals)[::-1][:n_suggestions]
        return [candidates[i].astype(np.float32) for i in top_idx]

    def update_best(self, rho: float) -> None:
        """Update the internal best robustness tracker."""
        if rho < self._rho_best:
            self._rho_best = rho

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _scaled_sobol(self, n: int) -> np.ndarray:
        """Generate n Sobol samples scaled to the parameter bounds."""
        raw = _sobol_sample(n, self.dim)  # [n, D] in [0,1]
        lo = self.bounds[:, 0]
        hi = self.bounds[:, 1]
        return (raw * (hi - lo) + lo).astype(np.float64)
