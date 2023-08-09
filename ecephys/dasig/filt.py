import dask.array as da
import numpy as np
import scipy.signal

from ecephys import npsig


def butter_bandpass(
    data: da.Array,
    lowcut: float,
    highcut: float,
    fs: float,
    order: int,
    time_axis=-1,
    plot: bool = True,
) -> da.Array:
    b, a = npsig.get_butter_bandpass_coefs(lowcut, highcut, fs, order, plot=plot)
    irlen = npsig.estimate_impulse_response_len(b, a, eps=1e-3)
    chunk_overlap = np.round(2 * irlen).astype(int)
    min_chunksize = 3 * chunk_overlap

    if data.chunksize[time_axis] < min_chunksize:
        raise ValueError(
            f"Chunks are too small for this filter's impulse response. Chunks should be at least {min_chunksize} samples at {fs} Hz. Please rechunk."
        )

    depth = dict(zip(range(data.ndim), [0] * data.ndim))
    if time_axis < 0:
        time_axis = (
            data.ndim + time_axis
        )  # Get the actual axis number, so we can use it as a dictionary key
    depth[time_axis] = chunk_overlap  # Key: Axis index, Value: axis depth

    def _filter(x):
        return scipy.signal.filtfilt(b, a, x, axis=time_axis)

    filtered = da.map_overlap(
        _filter, data, depth=depth, boundary="reflect", meta=data._meta
    )
    return filtered
