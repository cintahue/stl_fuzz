"""GlobalSurrogate: GPR (L1-level) + MLP (full 36D) condition surrogate models.

Two-tier design:
  - GPR: fast coarse prediction over 11D L1 features with uncertainty.
         Used for Bayesian Optimization candidate selection and pre-filter.
  - MLP: fine-grained prediction over full 36D (L1 + phase + L3) features.
         Used to guide active-learning inside the L2-L3 search loop.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("[GlobalSurrogate] scikit-learn not found; GPR disabled.")

from robostl.surrogate.data_buffer import DataBuffer


class _SurrogateMLP(nn.Module):
    def __init__(self, input_dim: int = 36, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class GlobalSurrogate:
    """Global condition surrogate: GPR (L1) + MLP (full 36D)."""

    def __init__(
        self,
        l1_dim: int = 11,
        full_dim: int = 36,
        device: Optional[str] = None,
    ) -> None:
        self.l1_dim = l1_dim
        self.full_dim = full_dim
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.gpr: Optional["GaussianProcessRegressor"] = None
        self._gpr_fitted = False

        self.mlp = _SurrogateMLP(input_dim=full_dim).to(self.device)
        self._mlp_fitted = False

        # Running normalization for MLP inputs
        self._x_mean = np.zeros(full_dim, dtype=np.float32)
        self._x_std = np.ones(full_dim, dtype=np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, buffer: DataBuffer, mlp_epochs: int = 150) -> None:
        """Train both GPR and MLP on all data in buffer."""
        X_l1, X_full, y = buffer.get_training_data()
        if len(y) == 0:
            return
        self._train_gpr(X_l1, y)
        self._train_mlp(X_full, y, epochs=mlp_epochs)

    def retrain(self, buffer: DataBuffer, mlp_epochs: int = 100) -> None:
        """Retrain (GPR is non-incremental; MLP partial retrain)."""
        self.train(buffer, mlp_epochs=mlp_epochs)

    def retrain_mlp(self, buffer: DataBuffer, mlp_epochs: int = 50) -> None:
        """MLP-only fast retrain."""
        _, X_full, y = buffer.get_training_data()
        if len(y) > 0:
            self._train_mlp(X_full, y, epochs=mlp_epochs)

    def gpr_predict(self, x_l1: np.ndarray) -> Tuple[float, float]:
        """Return (mean, std) prediction from GPR over L1 features."""
        if not self._gpr_fitted or self.gpr is None:
            return 0.0, 1.0
        try:
            x = np.asarray(x_l1, dtype=np.float64).reshape(1, -1)
            mu, sigma = self.gpr.predict(x, return_std=True)
            return float(mu[0]), float(sigma[0])
        except Exception:
            return 0.0, 1.0

    def mlp_predict(self, x_full: np.ndarray) -> float:
        """Single-sample MLP prediction."""
        if not self._mlp_fitted:
            return 0.0
        x = (np.asarray(x_full, dtype=np.float32) - self._x_mean) / self._x_std
        with torch.no_grad():
            t = torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)
            return float(self.mlp(t).item())

    def mlp_predict_batch(self, X_full: np.ndarray) -> np.ndarray:
        """Batch MLP prediction."""
        if not self._mlp_fitted:
            return np.zeros(len(X_full), dtype=np.float32)
        X_norm = (X_full.astype(np.float32) - self._x_mean) / self._x_std
        with torch.no_grad():
            t = torch.tensor(X_norm, dtype=torch.float32, device=self.device)
            return self.mlp(t).cpu().numpy()

    def get_top_k(
        self,
        x_l1: np.ndarray,
        k: int = 3,
        n_candidates: int = 5000,
        phase_bounds: Tuple[float, float] = (0.0, 1.0),
        l3_scale: float = 0.08,
    ) -> list[np.ndarray]:
        """Return k full-dim candidate vectors with lowest predicted rho.

        Randomly samples (phase, L3) while fixing x_l1, then returns the k
        candidates the MLP predicts as most dangerous.
        """
        if not self._mlp_fitted:
            return []
        phases = np.random.uniform(*phase_bounds, (n_candidates, 1)).astype(np.float32)
        l3 = np.random.uniform(-l3_scale, l3_scale, (n_candidates, 24)).astype(np.float32)
        l1_tiled = np.tile(np.asarray(x_l1, dtype=np.float32), (n_candidates, 1))
        X = np.concatenate([l1_tiled, phases, l3], axis=1)
        preds = self.mlp_predict_batch(X)
        top_idx = np.argsort(preds)[:k]
        return [X[i] for i in top_idx]

    @property
    def is_ready(self) -> bool:
        return self._gpr_fitted or self._mlp_fitted

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _train_gpr(self, X_l1: np.ndarray, y: np.ndarray) -> None:
        if not SKLEARN_AVAILABLE or len(y) == 0:
            return
        try:
            kernel = Matern(nu=2.5) + WhiteKernel(noise_level=1e-4)
            gpr = GaussianProcessRegressor(
                kernel=kernel,
                normalize_y=True,
                n_restarts_optimizer=3,
            )
            gpr.fit(X_l1.astype(np.float64), y.astype(np.float64))
            self.gpr = gpr
            self._gpr_fitted = True
        except Exception as exc:
            print(f"[GlobalSurrogate] GPR training failed: {exc}")
            self._gpr_fitted = False

    def _train_mlp(
        self, X_full: np.ndarray, y: np.ndarray, epochs: int = 150
    ) -> None:
        if len(y) == 0:
            return
        self._x_mean = X_full.mean(axis=0).astype(np.float32)
        self._x_std = np.maximum(X_full.std(axis=0), 1e-6).astype(np.float32)

        X_norm = (X_full.astype(np.float32) - self._x_mean) / self._x_std
        X_t = torch.tensor(X_norm, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y, dtype=torch.float32, device=self.device)

        optimizer = torch.optim.Adam(self.mlp.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()

        self.mlp.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            loss = loss_fn(self.mlp(X_t), y_t)
            loss.backward()
            optimizer.step()

        self.mlp.eval()
        self._mlp_fitted = True
