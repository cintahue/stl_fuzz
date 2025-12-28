from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class CMAESState:
    mean: np.ndarray
    sigma: float
    generation: int


class CMAES:
    def __init__(
        self,
        mean: np.ndarray,
        sigma: float,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        population_size: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.mean = np.array(mean, dtype=np.float32)
        self.sigma = float(sigma)
        self.bounds = bounds
        self.rng = np.random.default_rng(seed)

        self.n = int(self.mean.size)
        self.lambda_ = (
            int(population_size)
            if population_size is not None
            else 4 + int(3 * np.log(self.n))
        )
        self.mu = self.lambda_ // 2

        weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        weights = weights / np.sum(weights)
        self.weights = weights
        self.mu_eff = 1.0 / np.sum(self.weights**2)

        self.c_sigma = (self.mu_eff + 2.0) / (self.n + self.mu_eff + 5.0)
        self.d_sigma = 1.0 + 2.0 * max(
            0.0, np.sqrt((self.mu_eff - 1.0) / (self.n + 1.0)) - 1.0
        ) + self.c_sigma
        self.c_c = (4.0 + self.mu_eff / self.n) / (self.n + 4.0 + 2.0 * self.mu_eff / self.n)
        self.c1 = 2.0 / ((self.n + 1.3) ** 2 + self.mu_eff)
        self.c_mu = min(
            1.0 - self.c1,
            2.0 * (self.mu_eff - 2.0 + 1.0 / self.mu_eff) / ((self.n + 2.0) ** 2 + self.mu_eff),
        )

        self.chi_n = np.sqrt(self.n) * (
            1.0 - 1.0 / (4.0 * self.n) + 1.0 / (21.0 * self.n * self.n)
        )

        self.p_sigma = np.zeros(self.n, dtype=np.float32)
        self.p_c = np.zeros(self.n, dtype=np.float32)
        self.C = np.eye(self.n, dtype=np.float32)
        self.B = np.eye(self.n, dtype=np.float32)
        self.D = np.ones(self.n, dtype=np.float32)
        self.C_inv_sqrt = np.eye(self.n, dtype=np.float32)
        self.generation = 0
        self._last_x: Optional[np.ndarray] = None
        self._last_y: Optional[np.ndarray] = None
        self._update_eigensystem()

    def ask(self) -> np.ndarray:
        z = self.rng.standard_normal((self.lambda_, self.n)).astype(np.float32)
        y = (self.B @ (self.D[:, None] * z.T)).T
        x = self.mean + self.sigma * y
        if self.bounds is not None:
            low, high = self.bounds
            x = np.clip(x, low, high)
            y = (x - self.mean) / max(self.sigma, 1e-12)
        self._last_x = x
        self._last_y = y
        return x

    def tell(self, fitness: np.ndarray) -> CMAESState:
        if self._last_x is None or self._last_y is None:
            raise RuntimeError("ask() must be called before tell().")
        if fitness.shape[0] != self.lambda_:
            raise ValueError("Fitness size does not match population size.")

        idx = np.argsort(fitness)
        x = self._last_x[idx][: self.mu]
        y = self._last_y[idx][: self.mu]

        self.mean = np.sum(self.weights[:, None] * x, axis=0)
        y_w = np.sum(self.weights[:, None] * y, axis=0)

        self.p_sigma = (1.0 - self.c_sigma) * self.p_sigma + np.sqrt(
            self.c_sigma * (2.0 - self.c_sigma) * self.mu_eff
        ) * (self.C_inv_sqrt @ y_w)

        norm_p_sigma = float(np.linalg.norm(self.p_sigma))
        h_sigma = (
            norm_p_sigma
            / np.sqrt(1.0 - (1.0 - self.c_sigma) ** (2.0 * (self.generation + 1)))
            < (1.4 + 2.0 / (self.n + 1.0)) * self.chi_n
        )

        self.p_c = (1.0 - self.c_c) * self.p_c + (
            float(h_sigma)
            * np.sqrt(self.c_c * (2.0 - self.c_c) * self.mu_eff)
            * y_w
        )

        rank_one = np.outer(self.p_c, self.p_c)
        rank_mu = np.zeros_like(self.C)
        for wi, yi in zip(self.weights, y):
            rank_mu += wi * np.outer(yi, yi)

        self.C = (
            (1.0 - self.c1 - self.c_mu) * self.C
            + self.c1 * (rank_one + (1.0 - float(h_sigma)) * self.c_c * (2.0 - self.c_c) * self.C)
            + self.c_mu * rank_mu
        )

        self.sigma *= np.exp((self.c_sigma / self.d_sigma) * (norm_p_sigma / self.chi_n - 1.0))
        self.generation += 1
        self._update_eigensystem()

        return CMAESState(mean=self.mean.copy(), sigma=self.sigma, generation=self.generation)

    def _update_eigensystem(self) -> None:
        eigvals, eigvecs = np.linalg.eigh(self.C)
        eigvals = np.maximum(eigvals, 1e-20)
        self.B = eigvecs.astype(np.float32)
        self.D = np.sqrt(eigvals).astype(np.float32)
        inv_sqrt = 1.0 / self.D
        self.C_inv_sqrt = (self.B * inv_sqrt) @ self.B.T
