import numpy as np
import scipy.signal
import yasa


def decimate_timeseries(x: np.ndarray, q: int) -> np.ndarray:
    """
    x: (n_times, n_signals)
    q: Downsample factor
    """
    return scipy.signal.decimate(x, q=q, ftype="fir", axis=0)


def moving_transform(
    x: np.ndarray, fs: float, window: float, step: float, method: str
) -> np.ndarray:
    assert x.ndim == 2, "Data must be 2D."
    time_axis = 0
    channel_axis = 1

    mrms = np.zeros_like(x)
    for i in range(x.shape[channel_axis]):
        _, mrms[:, i] = yasa.moving_transform(
            x=x[:, i], sf=fs, window=window, step=step, method=method, interp=True
        )
    return mrms
