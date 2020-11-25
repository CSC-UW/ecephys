import numpy as np


def rms(data):
    """
    Computes root-mean-squared voltage of flattened array

    Input:
    -----
    data - numpy.ndarray

    Output:
    ------
    rms_value - float
    """
    return np.power(np.mean(np.power(data.astype("float32"), 2)), 0.5)


def median_subtract(sig):
    """Median subtraction on a per-channel basis.

    Parameters
    -----------
    sig : (n_samples,) or (n_samples, n_chans) array
        Time series.

    Returns
    -------
    sig: Same shape as input
        Time series with median subtraction applied.
    """
    return sig - np.median(sig, axis=0)


def mean_subtract(sig):
    return sig - np.nanmean(sig, axis=0)


def get_perievent_samples(event_time, time, fs, time_before, time_after):
    """Get the first and last sample indices in a perivent window.

    Parameters
    ----------
    event_time: float
        The time of the event.
    time: 1d array, (n_samples, )
        The sample times of the data.
    fs: float
        The sampling frequency of the data.
    time_before: float
        The amount of time to extract before the event.
    time_after: float
        The amount of time to extract after the event.

    Returns:
    (start_sample, end_sample): tuple of ints
        The first and last sample to extract around the event.
    """
    event_sample = (np.abs(time - event_time)).argmin()
    samples_before = np.int(time_before * fs)
    samples_after = np.int(time_after * fs)
    start_sample = event_sample - samples_before
    end_sample = event_sample + samples_after

    return (start_sample, end_sample)


def get_perievent_time(event_time, time, fs, time_before, time_after):
    start_sample, end_sample = get_perievent_samples(
        event_time, time, fs, time_before, time_after
    )
    return time[start_sample:end_sample]


def get_perievent_data(sigs, event_time, time, fs, time_before, time_after):
    start_sample, end_sample = get_perievent_samples(
        event_time, time, fs, time_before, time_after
    )
    evt_sigs = sigs[start_sample:end_sample, :]

    return evt_sigs


def _get_perievent_data(sigs, event_time, time, time_before, time_after):
    window_start_time = ripple_time - time_before
    window_end_time = ripple_time + time_after
    ripple_samples = np.logical_and(time >= window_start_time, time <= window_end_time)

    return sigs[ripple_samples, :], time[ripple_samples, :]