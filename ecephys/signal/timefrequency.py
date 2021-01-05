import numpy as np
from scipy.signal import spectrogram, cwt, morlet2
from neurodsp.spectral.checks import check_spg_settings
from neurodsp.spectral.utils import trim_spectrogram
from multiprocessing import Pool
from functools import partial

from ecephys.utils import ncols, all_arrays_equal
from ecephys.signal.utils import get_perievent_time, get_perievent_data


def compute_spectrogram_welch(
    sig,
    fs,
    window="hann",
    detrend="constant",
    nperseg=None,
    noverlap=None,
    f_range=None,
    t_range=None,
):
    """Compute spectrogram using Welch's method.

    Parameters
    -----------
    sig : (n_samples,)
        Time series.
    fs : float
        Sampling rate, in Hz.
    window : str or tuple or array_like, optional, default: 'hann'
        Desired window to use. See scipy.signal.get_window for a list of available windows.
        If array_like, the array will be used as the window and its length must be nperseg.
    detrend: str or function or False, optional
        Specifies how to detrend each segment. If detrend is a string, it is passed as the
        type argument to the detrend function. If it is a function, it takes a segment and
        returns a detrended segment. If detrend is False, no detrending is done.
        Defaults to ‘constant’, which is mean subtraction.
    nperseg : int, optional
        Length of each segment, in number of samples.
        If None, and window is str or tuple, is set to 1 second of data.
        If None, and window is array_like, is set to the length of the window.
    noverlap : int, optional
        Number of points to overlap between segments.
        If None, noverlap = nperseg // 8.
    f_range: list of [float, float]
        Frequency range to restrict to, as [f_low, f_high].
    t_range: list of [float, float]
        Time range to restrict to, as [t_low, t_high].

    Returns
    -------
    freqs : 1d array
        Frequencies at which the measure was calculated.
    spg_times: 1d array
        Segment times
    spg : (n_freqs, n_spg_times)
        Spectrogram of `sig`.
    """

    # Calculate the short time Fourier transform with signal.spectrogram
    nperseg, noverlap = check_spg_settings(fs, window, nperseg, noverlap)
    freqs, spg_times, spg = spectrogram(
        sig, fs, window, nperseg, noverlap, detrend=detrend
    )
    freqs, spg_times, spg = trim_spectrogram(freqs, spg_times, spg, f_range, t_range)

    return freqs, spg_times, spg


def parallel_spectrogram_welch(sig, fs, **kwargs):
    """Apply `compute_spectrogram_welch` to each channel in parallel.

    Parameters
    ----------
    sig: (n_samples, n_chans)
        The multichannel timeseries.
    fs: float
        The sampling frequency of the data.
    **kwargs: optional
        Keyword arguments passed to `compute_spectrogram_welch`.

    Returns:
    --------
    freqs : 1d array
        Frequencies at which the measure was calculated.
    spg_times: 1d array
        Segment times
    spg : (n_freqs, n_spg_times, n_chans)
        Spectrogram of `sig`.
    """
    assert ncols(sig) > 1, "Parallel spectrogram intended for multichannel data only."

    worker = partial(compute_spectrogram_welch, fs=fs, **kwargs)
    jobs = [x for x in sig.T]

    n_chans = ncols(sig)
    with Pool(n_chans) as p:
        freqs, spg_times, spg = zip(*p.map(worker, jobs))

    assert all_arrays_equal(
        freqs
    ), "Spectrogram frequecies must match for all channels."
    assert all_arrays_equal(spg_times), "Segment times must match for all channels."

    freqs = freqs[0]
    spg_times = spg_times[0]
    spg = np.dstack(spg)

    return freqs, spg_times, spg


def get_bandpower(freqs, spg_times, spg, f_range, t_range=None):
    """Get band-limited power from a spectrogram.

    Parameters
    ----------
    freqs: 1d array
        Frequencies at which spectral power was computed.
    spg_times: 1d array
        Times at which spectral power estimates are centered.
    spg: (n_freqs, n_spg_times)
        Spectrogram data
    f_range: list of [float, float]
        Frequency range to restrict to, as [f_low, f_high].
    t_range: list of [float, float]
        Time range to restrict to, as [t_low, t_high].
    """
    freqs, spg_times, spg = trim_spectrogram(freqs, spg_times, spg, f_range, t_range)
    bandpower = np.sum(spg, axis=0)

    return bandpower


def get_perievent_cwtm(evt_sig, fs, freq, normalize=False):
    w = 6
    widths = w * fs / (2 * freq * np.pi)
    cwtm = np.apply_along_axis(lambda sig: cwt(sig, morlet2, widths, w=w), 0, evt_sig)
    cwtm = np.mean(np.abs(cwtm), axis=2)  # Average across channels

    if normalize:
        cwtm = cwtm / (1 / freq)[:, None]

    return cwtm


def get_avg_perievent_cwtm(
    sig, times, fs, event_times, freq, time_before, time_after, norm=True
):

    # Determine how many samples are in each perievent window, so we can pre-allocate
    # space for the return.
    n_expected_samples = np.int(time_before * fs) + np.int(time_after * fs)

    # Filter out any events that don't have enough data on either side to satisfy the
    # perievent window.
    event_times = [
        event_time
        for event_time in event_times
        if get_perievent_time(event_time, times, fs, time_before, time_after).size
        == n_expected_samples
    ]

    # Pre-allocate space for the result
    result = np.zeros((freq.size, n_expected_samples, len(event_times)))

    # For each event, calculate the mean wavelet spectrogram across all channels.
    for i, event_time in enumerate(event_times):
        evt_sig = get_perievent_data(
            sig, event_time, times, fs, time_before, time_after
        )
        result[:, :, i] = get_perievent_cwtm(evt_sig, fs, freq)

    result = np.mean(result, axis=2)  # Average across events
    if norm:
        result = result / (1 / freq)[:, None]

    return result
