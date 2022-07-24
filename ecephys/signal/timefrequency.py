import numpy as np
from scipy.signal import spectrogram, cwt, morlet2
from multiprocessing import Pool
from functools import partial

from ecephys.utils import ncols, all_arrays_equal
from ecephys.signal.utils import get_perievent_time, get_perievent_data

# This function is taken directly from neurodsp.spectral.utils.
# We cannot use the neurodsp package, because a critical IBL library shadows the name.
def check_spg_settings(fs, window, nperseg, noverlap):
    """Check settings used for calculating spectrogram.
    Parameters
    ----------
    fs : float
        Sampling rate, in Hz.
    window : str or tuple or array_like
        Desired window to use. See scipy.signal.get_window for a list of available windows.
        If array_like, the array will be used as the window and its length must be nperseg.
    nperseg : int or None
        Length of each segment, in number of samples.
    noverlap : int or None
        Number of points to overlap between segments.
    Returns
    -------
    nperseg : int
        Length of each segment, in number of samples.
    noverlap : int
        Number of points to overlap between segments.
    """

    # Set the nperseg, if not provided
    if nperseg is None:

        # If the window is a string or tuple, defaults to 1 second of data
        if isinstance(window, (str, tuple)):
            nperseg = int(fs)
        # If the window is an array, defaults to window length
        else:
            nperseg = len(window)
    else:
        nperseg = int(nperseg)

    if noverlap is not None:
        noverlap = int(noverlap)

    return nperseg, noverlap


# This function is taken directly from neurodsp.spectral.utils.
# We cannot use the neurodsp package, because a critical IBL library shadows the name.
def trim_spectrogram(freqs, times, spg, f_range=None, t_range=None):
    """Extract a frequency or time range of interest from a spectrogram.
    Parameters
    ----------
    freqs : 1d array
        Frequency values for the spectrogram.
    times : 1d array
        Time values for the spectrogram.
    spg : 2d array
        Spectrogram, or time frequency representation of a signal.
        Formatted as [n_freqs, n_time_windows].
    f_range : list of [float, float]
        Frequency range to restrict to, as [f_low, f_high].
    t_range : list of [float, float]
        Time range to restrict to, as [t_low, t_high].
    Returns
    -------
    freqs_ext : 1d array
        Extracted frequency values for the power spectrum.
    times_ext : 1d array
        Extracted segment time values
    spg_ext : 2d array
        Extracted spectrogram values.
    Notes
    -----
    This function extracts frequency ranges >= f_low and <= f_high,
    and time ranges >= t_low and <= t_high. It does not round to below
    or above f_low and f_high, or t_low and t_high, respectively.
    Examples
    --------
    Trim the spectrogram of a simulated time series:
    >>> from neurodsp.sim import sim_combined
    >>> from neurodsp.timefrequency import compute_wavelet_transform
    >>> from neurodsp.utils.data import create_times, create_freqs
    >>> fs = 500
    >>> n_seconds = 10
    >>> times = create_times(n_seconds, fs)
    >>> sig = sim_combined(n_seconds, fs,
    ...                    components={'sim_powerlaw': {}, 'sim_oscillation' : {'freq': 10}})
    >>> freqs = create_freqs(1, 15)
    >>> mwt = compute_wavelet_transform(sig, fs, freqs)
    >>> spg = abs(mwt)**2
    >>> freqs_ext, times_ext, spg_ext = trim_spectrogram(freqs, times, spg,
    ...                                                  f_range=[8, 12], t_range=[0, 5])
    """

    # Initialize spg_ext, to define for case in which neither f_range nor t_range is defined
    spg_ext = spg

    # Restrict frequency range of the spectrogram
    if f_range is not None:
        f_mask = np.logical_and(freqs >= f_range[0], freqs <= f_range[1])
        freqs_ext = freqs[f_mask]
        spg_ext = spg_ext[f_mask, :]
    else:
        freqs_ext = freqs

    # Restrict time range of the spectrogram
    if t_range is not None:
        times_mask = np.logical_and(times >= t_range[0], times <= t_range[1])
        times_ext = times[times_mask]
        spg_ext = spg_ext[:, times_mask]
    else:
        times_ext = times

    return freqs_ext, times_ext, spg_ext


def single_spectrogram_welch(
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
    """Apply `_compute_spectrogram_welch` to each channel in parallel.

    Should also work fine for a single channel, as long as sig is 2D.
    But in that case, maybe you want to save the overhead and use
    single_spectrogram_welch directly...

    Parameters
    ----------
    sig: (n_samples, n_chans)
        The multichannel timeseries.
    fs: float
        The sampling frequency of the data.
    **kwargs: optional
        Keyword arguments passed to `_compute_spectrogram_welch`.

    Returns:
    --------
    freqs : 1d array
        Frequencies at which the measure was calculated.
    spg_times: 1d array
        Segment times
    spg : (n_freqs, n_spg_times, n_chans)
        Spectrogram of `sig`.
    """
    if ncols == 1:
        print("Using parallel_spectrogram_welch on single-channel data.")
        print("Maybe you want to use single_spectrogram_welch instead?")
    worker = partial(single_spectrogram_welch, fs=fs, **kwargs)
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
    if len(spg) > 1:
        spg = np.dstack(spg)
    else:
        spg = np.expand_dims(spg, axis=-1)

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
