import numpy as np
from numpy.linalg import pinv


def reconstruct(A: np.ndarray, y: np.ndarray) -> np.ndarray:
    return pinv(A) @ y
