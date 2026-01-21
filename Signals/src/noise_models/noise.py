from __future__ import annotations
import numpy as np


def add_gaussian_noise(y: np.ndarray, sigma: float, rng: np.random.Generator | None = None):
    rng = rng or np.random.default_rng()
    noise = rng.normal(0.0, sigma, size=y.shape)
    return y + noise, noise
