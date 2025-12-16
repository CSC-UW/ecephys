"""
Functions reproduced from the YASA package to avoid slow import times.

The functions in this module are adapted from YASA (Yet Another Spindle Algorithm)
https://github.com/raphaelvallat/yasa

Original code is licensed under the BSD-3-Clause License.
Copyright (c) 2018, Raphael Vallat

These functions are reproduced here to avoid the dependency on YASA, which has
extremely slow import times (>6 seconds) due to its many dependencies. Only the
minimal subset of functions needed by ecephys are included here.

Original YASA version: 0.6.x (2024)
"""

import numpy as np
from numba import jit
from scipy.interpolate import interp1d

__all__ = ["moving_transform", "merge_close", "detrend", "rms"]


#############################################################################
# NUMBA JIT UTILITY FUNCTIONS (from yasa.numba)
#############################################################################


@jit("float64(float64[:], float64[:])", nopython=True)
def _corr(x, y):
    """Fast Pearson correlation.

    Adapted from YASA (yasa.numba._corr)
    """
    mx, my = x.mean(), y.mean()
    xm2s, ym2s, r_num = 0, 0, 0
    for xi, yi in zip(x, y):
        xm = xi - mx
        ym = yi - my
        r_num += xm * ym
        xm2s += xm**2
        ym2s += ym**2
    r_d1 = np.sqrt(xm2s)
    r_d2 = np.sqrt(ym2s)
    r_den = r_d1 * r_d2
    if r_den == 0:
        return np.nan
    return r_num / r_den


@jit("float64(float64[:], float64[:])", nopython=True)
def _covar(x, y):
    """Fast Covariance.

    Adapted from YASA (yasa.numba._covar)
    """
    n = x.size
    mx, my = x.mean(), y.mean()
    cov = 0
    for i in range(n):
        xm = x[i] - mx
        ym = y[i] - my
        cov += xm * ym
    return cov / (n - 1)


@jit("float64(float64[:])", nopython=True)
def _rms(x):
    """Fast root mean square.

    Adapted from YASA (yasa.numba._rms)
    """
    n = x.size
    ms = 0
    for i in range(n):
        ms += x[i] ** 2
    ms /= n
    return np.sqrt(ms)


@jit("float64(float64[:], float64[:])", nopython=True)
def _slope_lstsq(x, y):
    """Slope of a 1D least-squares regression.

    Adapted from YASA (yasa.numba._slope_lstsq)
    """
    n_times = x.shape[0]
    sx2 = 0
    sx = 0
    sy = 0
    sxy = 0
    for j in range(n_times):
        sx2 += x[j] ** 2
        sx += x[j]
        sxy += x[j] * y[j]
        sy += y[j]
    den = n_times * sx2 - (sx**2)
    num = n_times * sxy - sx * sy
    if den == 0:
        return np.nan
    return num / den


@jit("float64[:](float64[:], float64[:])", nopython=True)
def _detrend(x, y):
    """Fast linear detrending.

    Adapted from YASA (yasa.numba._detrend)
    """
    slope = _slope_lstsq(x, y)
    intercept = y.mean() - x.mean() * slope
    return y - (x * slope + intercept)


#############################################################################
# PUBLIC API FUNCTIONS
#############################################################################


def rms(x):
    """Compute the root mean square of an array.

    Parameters
    ----------
    x : np.ndarray
        Input array (float64)

    Returns
    -------
    float
        Root mean square value

    Notes
    -----
    Adapted from YASA (yasa.numba._rms)
    """
    x = np.asarray(x, dtype=np.float64)
    return _rms(x)


def detrend(x, y):
    """Fast linear detrending.

    Parameters
    ----------
    x : np.ndarray
        X-coordinates (float64)
    y : np.ndarray
        Y-values to detrend (float64)

    Returns
    -------
    np.ndarray
        Detrended y values

    Notes
    -----
    Adapted from YASA (yasa.numba._detrend)
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return _detrend(x, y)


def merge_close(index, min_distance_ms, sf):
    """Merge events that are too close in time.

    Parameters
    ----------
    index : array_like
        Indices of supra-threshold events.
    min_distance_ms : int
        Minimum distance (ms) between two events to consider them as two
        distinct events
    sf : float
        Sampling frequency of the data (Hz)

    Returns
    -------
    f_index : array_like
        Filled (corrected) Indices of supra-threshold events

    Notes
    -----
    Adapted from YASA (yasa.others._merge_close)
    Original code imported from the Visbrain package.
    """
    # Convert min_distance_ms
    min_distance = min_distance_ms / 1000.0 * sf
    idx_diff = np.diff(index)
    condition = idx_diff > 1
    idx_distance = np.where(condition)[0]
    distance = idx_diff[condition]
    bad = idx_distance[np.where(distance < min_distance)[0]]
    # Fill gap between events separated with less than min_distance_ms
    if len(bad) > 0:
        fill = np.hstack(
            [np.arange(index[j] + 1, index[j + 1]) for i, j in enumerate(bad)]
        )
        f_index = np.sort(np.append(index, fill))
        return f_index
    else:
        return index


def moving_transform(
    x, y=None, sf=100, window=0.3, step=0.1, method="corr", interp=False
):
    """Moving transformation of one or two time-series.

    Parameters
    ----------
    x : array_like
        Single-channel data
    y : array_like, optional
        Second single-channel data (only used if method in ['corr', 'covar']).
    sf : float
        Sampling frequency.
    window : int
        Window size in seconds.
    step : int
        Step in seconds.
        A step of 0.1 second (100 ms) is usually a good default.
        If step == 0, overlap at every sample (slowest)
        If step == nperseg, no overlap (fastest)
        Higher values = higher precision = slower computation.
    method : str
        Transformation to use.
        Available methods are::

            'mean' : arithmetic mean of x
            'min' : minimum value of x
            'max' : maximum value of x
            'ptp' : peak-to-peak amplitude of x
            'prop_above_zero' : proportion of values of x that are above zero
            'rms' : root mean square of x
            'slope' : slope of the least-square regression of x (in a.u / sec)
            'corr' : Correlation between x and y
            'covar' : Covariance between x and y
    interp : boolean
        If True, a cubic interpolation is performed to ensure that the output
        has the same size as the input.

    Returns
    -------
    t : np.array
        Time vector, in seconds, corresponding to the MIDDLE of each epoch.
    out : np.array
        Transformed signal

    Notes
    -----
    Adapted from YASA (yasa.others.moving_transform)
    This function was inspired by the `transform_signal` function of the
    Wonambi package (https://github.com/wonambi-python/wonambi).
    """
    # Safety checks
    assert method in [
        "mean",
        "min",
        "max",
        "ptp",
        "rms",
        "prop_above_zero",
        "slope",
        "covar",
        "corr",
    ]
    x = np.asarray(x, dtype=np.float64)
    if y is not None:
        y = np.asarray(y, dtype=np.float64)
        assert x.size == y.size

    if step == 0:
        step = 1 / sf

    halfdur = window / 2
    n = x.size
    total_dur = n / sf
    last = n - 1
    idx = np.arange(0, total_dur, step)
    out = np.zeros(idx.size)

    # Define beginning, end and time (centered) vector
    beg = ((idx - halfdur) * sf).astype(int)
    end = ((idx + halfdur) * sf).astype(int)
    beg[beg < 0] = 0
    end[end > last] = last
    # Alternatively, to cut off incomplete windows (comment the 2 lines above)
    # mask = ~((beg < 0) | (end > last))
    # beg, end = beg[mask], end[mask]
    t = np.column_stack((beg, end)).mean(1) / sf

    if method == "mean":

        def func(x):
            return np.mean(x)

    elif method == "min":

        def func(x):
            return np.min(x)

    elif method == "max":

        def func(x):
            return np.max(x)

    elif method == "ptp":

        def func(x):
            return np.ptp(x)

    elif method == "prop_above_zero":

        def func(x):
            return np.count_nonzero(x >= 0) / x.size

    elif method == "slope":

        def func(x):
            times = np.arange(x.size, dtype=np.float64) / sf
            return _slope_lstsq(times, x)

    elif method == "covar":

        def func(x, y):
            return _covar(x, y)

    elif method == "corr":

        def func(x, y):
            return _corr(x, y)

    else:

        def func(x):
            return _rms(x)

    # Now loop over successive epochs
    if method in ["covar", "corr"]:
        for i in range(idx.size):
            out[i] = func(x[beg[i] : end[i]], y[beg[i] : end[i]])
    else:
        for i in range(idx.size):
            out[i] = func(x[beg[i] : end[i]])

    # Finally interpolate
    if interp and step != 1 / sf:
        f = interp1d(
            t, out, kind="cubic", bounds_error=False, fill_value=0, assume_sorted=True
        )
        t = np.arange(n) / sf
        out = f(t)

    return t, out
