import dask.array as da
import numpy as np
import mne.filter
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
    irlen = npsig.estimate_impulse_response_len(b, a, eps=1e-9)
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


def antialiasing_filter(
    data: da.Array,
    fs: float,
    q: int,
    time_axis: int = -1,
) -> da.Array:
    result_type = data.dtype
    assert (result_type == np.float64) or (
        result_type == np.float32
    ), "Data must be float64 or float32."
    assert (
        q < 13
    ), "It is recommended to call `decimate` multiple times for downsampling factors higher than 13. See scipy.signal.decimate docs."
    n = 8
    sos = scipy.signal.cheby1(n, 0.05, 0.8 / q, output="sos")
    sos = np.asarray(sos, dtype=result_type)

    b, a = scipy.signal.cheby1(n, 0.05, 0.8 / q, output="ba")
    irlen = npsig.estimate_impulse_response_len(b, a, eps=1e-9)
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
        return scipy.signal.sosfiltfilt(sos, x, axis=time_axis)

    filtered = da.map_overlap(
        _filter, data, depth=depth, boundary="reflect", meta=data._meta
    )
    return filtered


def mne_filter(
    data: da.Array,
    sfreq: float,
    l_freq: float,
    h_freq: float,
    filter_length="auto",
    l_trans_bandwidth="auto",
    h_trans_bandwidth="auto",
    method="fir",
    iir_params=None,
    phase="zero",
    fir_window="hamming",
    fir_design="firwin",
    pad="reflect_limited",  # This is in addition to dask padding. We cannot easily disable MNE's padding, unfortunately, so it is redunant (extra safe!), but the cost is minimal as long as your chunks are large enough.
) -> da.Array:
    # Fix options that are incomaptible with dask.
    picks = None
    n_jobs = 1
    copy = False
    assert data.ndim == 2, "Data must be 2D, channel x time."
    time_axis = 1

    original_dtype = data.dtype
    if not original_dtype in (np.float32, np.float64):
        raise ValueError("Data must be float32 or float64.")
    iir_params, method = mne.filter._check_method(method, iir_params)
    filt = mne.filter.create_filter(
        None,
        sfreq,
        l_freq,
        h_freq,
        filter_length,
        l_trans_bandwidth,
        h_trans_bandwidth,
        method,
        iir_params,
        phase,
        fir_window,
        fir_design,
    )
    if method in ("fir", "fft"):
        padlen = len(filt)

        def _filter(x):
            return mne.filter._overlap_add_filter(
                x.astype(np.float64), filt, None, phase, picks, n_jobs, copy, pad
            ).astype(original_dtype)

    else:
        padlen = filt["padlen"]

        def _filter(x):
            return mne.filter._iir_filter(
                x.astype(np.float64), filt, picks, n_jobs, copy, phase
            ).astype(original_dtype)

    chunk_overlap = np.round(2 * padlen).astype(int)
    min_chunksize = 3 * chunk_overlap
    if data.chunksize[time_axis] < min_chunksize:
        raise ValueError(
            f"Chunks are too small for this filter's approximate impulse response. Chunks should be at least {min_chunksize} samples at {sfreq} Hz. Please rechunk."
        )

    depth = dict(zip(range(data.ndim), [0] * data.ndim))
    depth[time_axis] = chunk_overlap  # Key: Axis index, Value: axis depth

    filtered = da.map_overlap(
        _filter, data, depth=depth, boundary="reflect", meta=data._meta
    )
    return filtered
