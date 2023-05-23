import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from .. import utils
from ..signal import event_detection as evd

logger = logging.getLogger(__name__)


def plot_filter_response(fs, w, h, title):
    "Utility function to plot response functions"
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(0.5 * fs * w / np.pi, 20 * np.log10(np.abs(h)))
    ax.set_ylim(-40, 5)
    ax.set_xlim(0, 0.5 * fs)
    ax.grid(True)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Gain (dB)")
    ax.set_title(title)


def ripple_bandpass_filter(sampling_frequency, ripple_band=(125, 250), plot=False):
    """
    Notes
    ------
    Based on Eric Denovellis' ripple detection package [1].
    [1] https://github.com/Eden-Kramer-Lab/ripple_detection
    """
    ORDER = 101
    nyquist = 0.5 * sampling_frequency
    TRANSITION_BAND = 25
    desired = [
        0,
        ripple_band[0] - TRANSITION_BAND,
        ripple_band[0],
        ripple_band[1],
        ripple_band[1] + TRANSITION_BAND,
        nyquist,
    ]
    taps = signal.remez(ORDER, desired, [0, 1, 0], fs=sampling_frequency)
    if plot:
        w, h = signal.freqz(taps, [1], worN=1024)
        plot_filter_response(sampling_frequency, w, h, "Ripple filter response.")
    return taps, 1.0


def apply_ripple_filter(sig, fs, ripple_band=(125, 250), plot_filter=False):
    """Apply ripple filter to multichannel lfp data.

    Parameters
    ----------
    sig: (n_samples, n_chans)
        The data to filter.
    fs: float
        The sampling frequency of the data.

    Returns
    -------
    filtered_sig: (n_samples, n_chans)
        The filtered signal.

    Notes
    ------
    Based on Eric Denovellis' ripple detection package [1].
    [1] https://github.com/Eden-Kramer-Lab/ripple_detection
    """
    filter_numerator, filter_denominator = ripple_bandpass_filter(
        fs, ripple_band, plot_filter
    )
    is_nan = np.any(np.isnan(sig), axis=-1)
    filtered_sig = np.full_like(sig, np.nan)
    filtered_sig[~is_nan] = signal.filtfilt(
        filter_numerator, filter_denominator, sig[~is_nan], axis=0
    )

    return filtered_sig


def get_peak_info(sig, evts):
    """Get properties of each SPW peak.

    Parameters
    ==========
    sig: (time,) DataArray
        The signal to extract peak amplitudes, times, and channels from.
        Probably the series used for event detection.
    evts: DataFrame
        The events, with each event's start and end times.
    """
    evts = evts.copy()
    if len(evts) == 0:
        return evts

    def _get_peak_info(spw):
        spw_sig = sig.sel(time=slice(spw.start_time, spw.end_time))
        peak = spw_sig[spw_sig.argmax()]
        return peak.item(), peak.time.item(), peak.channel.item()

    info = list(map(_get_peak_info, evts.itertuples()))
    evts[["pk_amp", "pk_time", "pk_chan_id"]] = info
    return evts


def get_coarse_detection_chans(center, nCoarse, xrObj):
    """Given a channel around which to detect events, get the neighboring channels.

    Parameters:
    ===========
    center: int
        The channel around which to detect events.
    nCoarse: int
        An odd integer, indiciating the number of neighboring channels (inclusive) to use for detecting events.
    xrObj: xr.DataArray
        The xarray signals with channels for which you have data to detect.
    """
    assert nCoarse % 2, "Must use an odd number of of detection channels."
    nChansAbove = nChansBelow = nCoarse // 2
    ix = utils.find_nearest(xrObj.channel.values, center)

    if (ix - nChansBelow) < 0:
        ix = nChansBelow
        logger.warning("Requested channels outside the bounds of your data.")
    if (ix + nChansAbove + 1) > xrObj.channel.size:
        ix = xrObj.channel.size - nChansAbove - 1
        logger.warning("Requested channels outside the bounds of your data.")
    chans = xrObj.isel(
        channel=slice(ix - nChansBelow, ix + nChansAbove + 1)
    ).channel.values

    return chans


def _get_epoched_ripple_density(
    ripples, recording_start_time, recording_end_time, epoch_length
):
    epochs = evd.get_epoched_event_density(
        ripples, recording_start_time, recording_end_time, epoch_length
    )
    return epochs.rename(
        columns={"event_count": "n_ripples", "event_density": "ripple_density"}
    )
