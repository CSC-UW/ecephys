# -*- coding: utf-8 -*-

"""
Derive EMG from LFP through correlation of highfrequency activity.

This code is basically a Python translation and readaptation of the
bz_EMGFromLFP.m function from Buzsaki's lab.
(`https://github.com/buzsakilab/buzcode/blob/master/detectors/bz_EMGFromLFP.m`).
Based on Erik Schomburg's and code and work, published in `Theta Phase
Segregation of Input-Specific Gamma Patterns in Entorhinal-Hippocampal Networks,
Schomburg et al., Neuron 2014.


Tom Bugnon, 01/2020
"""

import numpy as np
import scipy.signal
from numba import njit


def compute(
    lfp,
    sf,
    target_sf,
    window_size,
    wp,
    ws,
    gpass=1,
    gstop=20,
    ftype="butter",
    method="both",
):
    """Derive EMG from LFP.

    Compute av. correlation across channel pairs in sliding windows.

    Args:
        lfp: (nchan x nsamples) lfp data array
        sf: Sampling frequency of input LFP array (Hz)
    Kwargs:
        target_sf: Sampling frequency of output "EMG" array (Hz)
        window_size: Duration (in ms) of each window during which average
            correlation is computed
        wp, ws, gpass, gstop, ftype: passed to
            scipy.signal.iirdesign
        method: EMG estimator(s) (see docs/scientific_conceptual/
            synthetic_emg_methods.md):
            - "per_window": exact per-window mean pairwise Pearson correlation
              (the Buzsaki/Schomburg method), amplitude-independent, bounded.
            - "global": faster global-normalization approximation, amplitude-
              weighted and not bounded to [-1, 1].
            - "both" (default): compute both. The expensive band-pass filter is
              shared, so this costs only marginally more than a single method.

    Returns:
        For a single method: an ``(1 x nSteps)`` array. For ``method="both"``: a
        ``dict`` mapping each method name to its ``(1 x nSteps)`` array. Sampling
        frequency is ``target_sf``.
    """
    estimators = {"per_window": _compute_av_corr, "global": _compute_global_corr}
    if method == "both":
        wanted = ("per_window", "global")
    elif method in estimators:
        wanted = (method,)
    else:
        raise ValueError(
            f"Unknown EMG method {method!r}; use 'per_window', 'global', or 'both'."
        )

    print(
        f"Filtering LFP with wp={wp}, ws={ws}, gpass={gpass}, gstop={gstop},"
        f"filter type={ftype}"
    )
    lfp_filt = _iirfilt(lfp, wp, ws, gpass, gstop, ftype="butter", sf=sf)
    print(f"Computing EMG from filtered LFP (method={method!r})...")
    print(
        f"target sf = {target_sf}, window size = {window_size}, LFP sf={sf},"
        f" LFP nchans = {lfp_filt.shape[0]}"
    )
    # Filter once above; run each requested estimator on the shared filtered LFP.
    results = {m: estimators[m](lfp_filt, sf, target_sf, window_size) for m in wanted}
    print("Done!")
    return results if method == "both" else results[method]


def _iirfilt(data, wp, ws, gpass, gstop, ftype="butter", sf=None):
    """Filter `data` along last dimension using an iir filter."""

    # Check input values to avoid https://github.com/scipy/scipy/issues/11559
    wp_check, ws_check = np.array(wp), np.array(ws)
    if sf is not None:
        wp_check, ws_check = wp_check / (sf / 2), ws_check / (sf / 2)
    if not (
        (np.all(wp_check > 0))
        & (np.all(wp_check < 1))
        & (np.all(ws_check > 0))
        & (np.all(ws_check < 1))
    ):
        raise ValueError("Digital filter critical frequencies must be 0 < Wn < 1")

    sos = scipy.signal.iirdesign(
        wp_check,
        ws_check,
        gpass,
        gstop,
        ftype=ftype,
        fs=None,  # Don't normalize (again) by Nyquist
        analog=False,
        output="sos",
    )

    return scipy.signal.sosfilt(sos, data)


# A window's per-channel variance below this fraction of the channel's global
# variance is treated as constant input -> NaN, matching scipy.stats.pearsonr.
# Sits ~orders of magnitude above running-sum rounding noise yet far below any
# genuine window variance.
_DEGENERATE_RTOL = 1e-8


def _window_bounds(n_samps, data_sf, target_sf, window_size):
    """Window grid and per-window sample bounds.

    Returns (centers, a, b_excl, window_n_samps) where each window covers the
    half-open interval ``[a[s], b_excl[s])``. The ``+1`` inclusive end is clamped
    by ``n_samps`` on the final window(s), so the last window(s) are one sample
    shorter than interior windows.
    """
    window_n_samps = int(window_size * data_sf)
    win_timestamps = np.arange(0, n_samps / data_sf, 1 / target_sf)
    centers = (win_timestamps * data_sf).astype(int)
    win_start = np.maximum(0, (centers - window_n_samps / 2).astype(int))
    win_end = np.minimum(n_samps, (centers + window_n_samps / 2).astype(int))
    a = win_start
    b_excl = np.minimum(win_end + 1, n_samps)
    return centers, a, b_excl, window_n_samps


@njit(cache=True, fastmath=False)
def _av_corr_kernel(data, a, b, pi, pj, gvar, rtol, recompute_every):
    """Mean pairwise Pearson correlation per window via incremental sliding sums.

    For each channel pair, running sums (Sx, Sy, Sxx, Syy, Sxy) are advanced as
    the window steps forward by adding entering / removing leaving samples
    (O(n_samps) per pair). Sums are recomputed from scratch every
    ``recompute_every`` windows to bound floating-point drift. Pairs touching a
    (near-)constant channel in a window contribute NaN, matching
    ``scipy.stats.pearsonr``. Returns the summed correlation per window (caller
    divides by the number of pairs).
    """
    n_pairs = pi.size
    n_win = a.size
    acc = np.zeros(n_win)
    for p in range(n_pairs):
        ci = pi[p]
        cj = pj[p]
        xi = data[ci]
        xj = data[cj]
        thr_i = rtol * gvar[ci]
        thr_j = rtol * gvar[cj]
        dead = gvar[ci] == 0.0 or gvar[cj] == 0.0
        Sx = Sy = Sxx = Syy = Sxy = 0.0
        cur_a = 0
        cur_b = 0
        for s in range(n_win):
            na = a[s]
            nb = b[s]
            if s % recompute_every == 0:
                Sx = Sy = Sxx = Syy = Sxy = 0.0
                for t in range(na, nb):
                    vi = xi[t]
                    vj = xj[t]
                    Sx += vi
                    Sy += vj
                    Sxx += vi * vi
                    Syy += vj * vj
                    Sxy += vi * vj
                cur_a = na
                cur_b = nb
            else:
                while cur_b < nb:
                    vi = xi[cur_b]
                    vj = xj[cur_b]
                    Sx += vi
                    Sy += vj
                    Sxx += vi * vi
                    Syy += vj * vj
                    Sxy += vi * vj
                    cur_b += 1
                while cur_b > nb:
                    cur_b -= 1
                    vi = xi[cur_b]
                    vj = xj[cur_b]
                    Sx -= vi
                    Sy -= vj
                    Sxx -= vi * vi
                    Syy -= vj * vj
                    Sxy -= vi * vj
                while cur_a < na:
                    vi = xi[cur_a]
                    vj = xj[cur_a]
                    Sx -= vi
                    Sy -= vj
                    Sxx -= vi * vi
                    Syy -= vj * vj
                    Sxy -= vi * vj
                    cur_a += 1
                while cur_a > na:
                    cur_a -= 1
                    vi = xi[cur_a]
                    vj = xj[cur_a]
                    Sx += vi
                    Sy += vj
                    Sxx += vi * vi
                    Syy += vj * vj
                    Sxy += vi * vj
            n = nb - na
            if n < 2:
                acc[s] += np.nan
                continue
            cov = Sxy - Sx * Sy / n
            varx = Sxx - Sx * Sx / n
            vary = Syy - Sy * Sy / n
            if dead or varx <= thr_i * n or vary <= thr_j * n:
                acc[s] += np.nan
            else:
                r = cov / np.sqrt(varx * vary)
                if r > 1.0:
                    r = 1.0
                elif r < -1.0:
                    r = -1.0
                acc[s] += r
    return acc


def _compute_av_corr(data, data_sf, target_sf, window_size, recompute_every=4096):
    """Compute av. correlation across channel pairs in sliding windows.

    Each output sample is the mean Pearson correlation, across all channel pairs,
    of the signals within a window of ``window_size * data_sf`` samples centered
    on that output sample. Computed with an incremental sliding-window kernel that
    visits each sample O(1) times rather than recomputing heavily overlapping
    windows (see :func:`_av_corr_kernel`).

    Args:
        data: (nchan x nsamples) data array
        data_sf: Sampling frequency of data array (Hz)
    Kwargs:
        target_sf: Desired sampling frequency of output time course
            (1/windowStep) (Hz)
        window_size: Duration (in ms) of each window during which average
            correlation is computed
        recompute_every: Window stride between exact recomputations of the
            running sums (bounds floating-point drift).

    Returns:
        corrData: (1 x nSteps) data array. Sampling frequency is `targetsf`
    """
    data = np.ascontiguousarray(data, dtype=np.float64)
    n_chans, n_samps = data.shape
    assert n_chans > 1

    # Per-channel global variance sets the scale for the (near-)constant-channel
    # degeneracy test. Mean-centering improves conditioning and is correlation-
    # invariant.
    gvar = data.var(axis=1)
    data = data - data.mean(axis=1, keepdims=True)

    _, a, b_excl, _ = _window_bounds(n_samps, data_sf, target_sf, window_size)
    pairs = [(i, j) for i in range(n_chans) for j in range(i + 1, n_chans)]
    n_pairs = len(pairs)
    pi = np.array([p[0] for p in pairs], dtype=np.int64)
    pj = np.array([p[1] for p in pairs], dtype=np.int64)

    acc = _av_corr_kernel(
        data,
        a.astype(np.int64),
        b_excl.astype(np.int64),
        pi,
        pj,
        gvar,
        _DEGENERATE_RTOL,
        np.int64(recompute_every),
    )
    return (acc / n_pairs).reshape(1, -1)


def _compute_global_corr(data, data_sf, target_sf, window_size):
    """Approximate mean pairwise correlation via global normalization.

    Faster/simpler alternative to :func:`_compute_av_corr`: each channel is
    z-scored once over the whole segment (rather than re-normalized per window),
    so the windowed mean of the summed pairwise products of z-scored signals
    reduces to a single boxcar moving average. O(n_chans * n_samps), no per-pair
    or per-window loop.

    WARNING -- this is a methodological approximation, not an exact reformulation.
    Because normalization is global rather than per-window, the estimator is
    AMPLITUDE-WEIGHTED: windows with higher high-frequency power yield larger
    values, and the output is NOT bounded to [-1, 1]. Validated against the exact
    estimator on real tetrode LFP: it preserves shape/ranking well (Spearman
    ~0.97) but not values (Pearson ~0.87), and responds spuriously (~6.7x) to
    amplitude changes at fixed correlation. See
    docs/scientific_conceptual/synthetic_emg_methods.md.
    """
    data = np.asarray(data, dtype=np.float64)
    n_chans, n_samps = data.shape
    assert n_chans > 1

    sd = data.std(axis=1, keepdims=True)
    sd = np.where(sd == 0.0, np.nan, sd)  # dead channel -> NaN output
    z = (data - data.mean(axis=1, keepdims=True)) / sd
    S = z.sum(axis=0)
    # Per-sample sum over channel pairs of z_i * z_j = (S^2 - sum_i z_i^2) / 2.
    P = (S * S - np.square(z).sum(axis=0)) / 2.0

    cumP = np.empty(n_samps + 1)
    cumP[0] = 0.0
    np.cumsum(P, out=cumP[1:])

    _, a, b_excl, _ = _window_bounds(n_samps, data_sf, target_sf, window_size)
    N = (b_excl - a).astype(np.float64)
    n_pairs = n_chans * (n_chans - 1) / 2
    win_mean = (cumP[b_excl] - cumP[a]) / N
    return (win_mean / n_pairs).reshape(1, -1)
