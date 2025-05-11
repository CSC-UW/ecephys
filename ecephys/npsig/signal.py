import numpy as np
import scipy.fftpack
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
    channel_axis = 1

    mrms = np.zeros_like(x)
    for i in range(x.shape[channel_axis]):
        _, mrms[:, i] = yasa.moving_transform(
            x=x[:, i], sf=fs, window=window, step=step, method=method, interp=True
        )
    return mrms


def hilbert(x: np.ndarray) -> np.ndarray:
    """
    Compute the analytic signal of `x` using the Hilbert transform.

    x: (n_times, n_signals)
    """
    ns = x.shape[0]
    nfast = scipy.fftpack.next_fast_len(ns)
    # For unclear reasons, this is faster than not using the N= kwarg, or than forgoing
    # transpose and using axis=0.
    return scipy.signal.hilbert(x, N=nfast, axis=0)[:ns, :]
