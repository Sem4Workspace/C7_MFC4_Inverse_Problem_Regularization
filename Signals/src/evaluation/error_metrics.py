import numpy as np
import math


def mse(x_true: np.ndarray, x_est: np.ndarray) -> float:
    return float(np.mean((x_true - x_est) ** 2))


def psnr(x_true: np.ndarray, x_est: np.ndarray) -> float:
    m = mse(x_true, x_est)
    if m == 0:
        return math.inf
    max_val = np.max(np.abs(x_true))
    return 20 * math.log10(max_val) - 10 * math.log10(m)


def relative_error(x_true: np.ndarray, x_est: np.ndarray) -> float:
    num = np.linalg.norm(x_true - x_est)
    den = np.linalg.norm(x_true)
    return float(num / den)
