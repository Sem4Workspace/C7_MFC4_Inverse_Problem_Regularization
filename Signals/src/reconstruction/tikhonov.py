import numpy as np
from numpy.linalg import svd


def reconstruct(A: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    U, s, Vt = svd(A, full_matrices=False)
    filt = s / (s**2 + lam**2)
    return (Vt.T * filt) @ (U.T @ y)
