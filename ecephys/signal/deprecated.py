import numpy as np
from scipy.signal import cwt, morlet2

from ecephys.signal.utils import get_perievent_time, get_perievent_data


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
