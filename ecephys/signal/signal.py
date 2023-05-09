import numpy as np
from scipy import signal


def decimate_timeseries(x: np.ndarray, q: int) -> np.ndarray:
    """
    x: (n_times, n_signals)
    q: Downsample factor
    """
    return signal.decimate(x, q=q, ftype="fir", axis=0)
