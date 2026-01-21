import numpy as np

def sinusoid(t: np.ndarray) -> np.ndarray:
    return np.sin(2 * np.pi * t) + 0.5 * np.sin(6 * np.pi * t)


def multisine(t: np.ndarray, freqs=(2, 5, 9), amps=(1.0, 0.6, 0.3)) -> np.ndarray:
    out = np.zeros_like(t)
    for f, a in zip(freqs, amps):
        out += a * np.sin(2 * np.pi * f * t)
    return out


def piecewise(t: np.ndarray) -> np.ndarray:
    out = np.zeros_like(t)
    thirds = len(t) // 3
    out[:thirds] = 1.0
    out[thirds:2 * thirds] = -0.5
    out[2 * thirds:] = 0.7
    return out
