import numpy as np


def tikhonov_filter(s: np.ndarray, lam: float) -> np.ndarray:
    return s / (s**2 + lam**2)


def tsvd_filter(s: np.ndarray, k: int) -> np.ndarray:
    filt = np.zeros_like(s)
    filt[:k] = 1.0 / s[:k]
    return filt
