import numpy as np
from scipy import signal
import ripple_detection.core as rdc
from ..signal import event_detection as evd


def apply_ripple_filter(sig, fs):
    """Apply 150-250Hz ripple filter to multichannel lfp data.

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
    filter_numerator, filter_denominator = rdc.ripple_bandpass_filter(fs)
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

    def _get_peak_info(spw):
        spw_sig = sig.sel(time=slice(spw.start_time, spw.end_time))
        peak = spw_sig[spw_sig.argmax()]
        return peak.item(), peak.time.item(), peak.channel.item()

    info = list(map(_get_peak_info, evts.itertuples()))
    evts[["peak_amplitude", "peak_time", "peak_channel"]] = info
    return evts


def get_coarse_detection_chans(center, n_coarse, chans):
    """Given a channel around which to detect events, get the neighboring channels.

    Parameters:
    ===========
    center: int
        The channel around which to detect events.
    n_coarse: int
        An odd integer, indiciating the number of neighboring channels (inclusive) to use for detecting events.
    chans: np.array
        The channels for which you have data to detect.
    """
    assert n_coarse % 2, "Must use an odd number of of detection channels."

    idx = chans.index(center)
    first = idx - n_coarse // 2
    last = idx + n_coarse // 2 + 1

    assert first >= 0, "Cannot detect events outside the bounds of your data."
    assert last < len(chans), "Cannot detect events outside the bounds of your data."

    return chans[first:last]


def _get_epoched_ripple_density(
    ripples, recording_start_time, recording_end_time, epoch_length
):
    epochs = evd.get_epoched_event_density(
        ripples, recording_start_time, recording_end_time, epoch_length
    )
    return epochs.rename(
        columns={"event_count": "n_ripples", "event_density": "ripple_density"}
    )
