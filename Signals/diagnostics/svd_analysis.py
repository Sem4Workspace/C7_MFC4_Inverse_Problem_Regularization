import numpy as np
from numpy.linalg import svd


def singular_values(A: np.ndarray) -> np.ndarray:
    _, s, _ = svd(A, full_matrices=False)
    return s


def condition_number(A: np.ndarray) -> float:
    s = singular_values(A)
    return float(s[0] / s[-1])
