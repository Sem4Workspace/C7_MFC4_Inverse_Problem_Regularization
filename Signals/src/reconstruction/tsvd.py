import numpy as np
from numpy.linalg import svd


def reconstruct(A: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    U, s, Vt = svd(A, full_matrices=False)
    filt = np.zeros_like(s)
    filt[:k] = 1.0 / s[:k]
    return (Vt.T * filt) @ (U.T @ y)
