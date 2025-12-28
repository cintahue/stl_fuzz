from __future__ import annotations

from pathlib import Path
import numpy as np
import torch


class TorchScriptPolicy:
    def __init__(self, path: Path, device: str = "cpu") -> None:
        self.path = Path(path)
        self.device = torch.device(device)
        self.module = torch.jit.load(str(self.path), map_location=self.device)
        self.module.eval()

    def act(self, obs: np.ndarray) -> np.ndarray:
        obs_tensor = torch.from_numpy(obs).to(self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.module(obs_tensor).cpu().numpy().squeeze()
        return action.astype(np.float32)
