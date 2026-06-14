"""Shared utilities for metric computations."""

import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over a 1-D numpy array."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
