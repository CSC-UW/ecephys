import numpy as np
import pandas as pd
from ecephys.helpers import unnest_df
from ecephys.scoring import mask_times, filter_states
from ripple_detection.core import (
    exclude_close_events,
    exclude_movement,
    _extend_segment,
    gaussian_smooth,
    get_envelope,
    ripple_bandpass_filter,
    segment_boolean_series,
)
from scipy.signal import filtfilt
from scipy.stats import zscore

# Based on Eric Denovellis' ripple detection package [1].
# [1] https://github.com/Eden-Kramer-Lab/ripple_detection


def make_speed_vector(H, states, times):
    """Create a fictitious speed vector such that the animal is stationary during
    the states during which you want to detect ripples, and moving above the detection
    threshold at all other times.

    Parameters
    ----------
    H: pandas.DataFrame
        Hypnogram, with 'state', 'start_time', and 'end_time' fields.
    states: list of str
        States during which you want to detect ripples.
    times: (n_samples), float
        Time of each LFP sample used for detection.

    Returns
    -------
    speed: (n_samples,)
        Each timepoint is 0 if the animal was in one of the states provided,
        else np.inf.
    """
    speed = np.full_like(times, np.inf)
    speed[mask_times(H, states, times)] = 0

    return speed


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
    """
    filter_numerator, filter_denominator = ripple_bandpass_filter(fs)
    is_nan = np.any(np.isnan(sig), axis=-1)
    filtered_sig = np.full_like(sig, np.nan)
    filtered_sig[~is_nan] = filtfilt(
        filter_numerator, filter_denominator, sig[~is_nan], axis=0
    )

    return filtered_sig


def extend_detection_threshold_to_boundaries(
    is_above_boundary_threshold,
    is_above_detection_threshold,
    time,
    minimum_duration=0.015,
):
    """Extract segments above threshold if they remain above the threshold
    for a minimum amount of time and extend them to a boundary threshold.

    Parameters
    ----------
    is_above_boundary_threshold : ndarray, shape (n_time,)
        Time series indicator function specifying when the
        time series is above the boundary threshold.
    is_above_detection_threshold : ndarray, shape (n_time,)
        Time series indicator function specifying when the
        time series is above the the detection_threshold.
    time : ndarray, shape (n_time,)

    Returns
    -------
    candidate_ripple_times : list of 2-element tuples
        Each tuple is the start and end time of the candidate ripple.
    """
    is_above_detection_threshold = pd.Series(is_above_detection_threshold, index=time)
    is_above_boundary_threshold = pd.Series(is_above_boundary_threshold, index=time)
    above_boundary_segments = segment_boolean_series(
        is_above_boundary_threshold, minimum_duration=minimum_duration
    )
    above_detection_segments = segment_boolean_series(
        is_above_detection_threshold, minimum_duration=minimum_duration
    )
    return sorted(_extend_segment(above_detection_segments, above_boundary_segments))


def _threshold_by_zscore(
    data,
    time,
    minimum_duration=0.015,
    detection_zscore_threshold=2,
    boundary_zscore_threshold=0,
):
    """Standardize the data and determine whether it is above a given
    number.

    Parameters
    ----------
    data : array_like, shape (n_time,)
    detection_zscore_threshold : int, optional
    boundary_zscore_threshold: int, optional

    Returns
    -------
    candidate_ripple_times : pandas Dataframe
    """
    zscored_data = zscore(data)
    is_above_boundary_threshold = zscored_data >= boundary_zscore_threshold
    is_above_detection_threshold = zscored_data >= detection_zscore_threshold

    return extend_detection_threshold_to_boundaries(
        is_above_boundary_threshold,
        is_above_detection_threshold,
        time,
        minimum_duration=minimum_duration,
    )


def generalized_Kay_ripple_detector(
    time,
    filtered_lfps,
    speed,
    sampling_frequency,
    speed_threshold=4.0,
    minimum_duration=0.015,
    detection_zscore_threshold=2.0,
    boundary_zscore_threshold=0.0,
    smoothing_sigma=0.004,
    close_ripple_threshold=0.0,
):
    """Find start and end times of sharp wave ripple events (150-250 Hz)
     based on the Kay method, but allowing boundaries to be defined by Z-scores
     rather than the mean.

     Parameters
     ----------
     time : array_like, shape (n_time,)
     filtered_lfps : array_like, shape (n_time, n_signals)
         Bandpass filtered time series of electric potentials in the ripple band
     speed : array_like, shape (n_time,)
         Running speed of animal
     sampling_frequency : float
         Number of samples per second.
     speed_threshold : float, optional
         Maximum running speed of animal for a ripple
     minimum_duration : float, optional
         Minimum time the z-score has to stay above threshold to be
         considered a ripple. The default is given assuming time is in
         units of seconds.
     detection_zscore_threshold : float, optional
         Number of standard deviations the ripple power must exceed to
         be considered a ripple
    boundary_zscore_threshold : float, optional
         Number of standard deviations the ripple power must drop
         below to define the ripple start or end time.
     smoothing_sigma : float, optional
         Amount to smooth the time series over time. The default is
         given assuming time is in units of seconds.
     close_ripple_threshold : float, optional
         Exclude ripples that occur within `close_ripple_threshold` of a
         previously detected ripple.

     Returns
     -------
     ripple_times : pandas DataFrame
    """
    filtered_lfps = np.asarray(filtered_lfps)
    not_null = np.all(pd.notnull(filtered_lfps), axis=1) & pd.notnull(speed)
    filtered_lfps, speed, time = (
        filtered_lfps[not_null],
        speed[not_null],
        time[not_null],
    )

    filtered_lfps = get_envelope(filtered_lfps)
    combined_filtered_lfps = np.sum(filtered_lfps ** 2, axis=1)
    combined_filtered_lfps = gaussian_smooth(
        combined_filtered_lfps, smoothing_sigma, sampling_frequency
    )
    combined_filtered_lfps = np.sqrt(combined_filtered_lfps)

    candidate_ripple_times = _threshold_by_zscore(
        combined_filtered_lfps,
        time,
        minimum_duration,
        detection_zscore_threshold,
        boundary_zscore_threshold,
    )

    ripple_times = exclude_movement(
        candidate_ripple_times, speed, time, speed_threshold=speed_threshold
    )
    ripple_times = exclude_close_events(ripple_times, close_ripple_threshold)
    index = pd.Index(np.arange(len(ripple_times)) + 1, name="ripple_number")
    return pd.DataFrame(ripple_times, columns=["start_time", "end_time"], index=index)


def get_ripple_rms_features(time, filtered_lfps, ripple_times):
    """Get features of the root-mean-square of LFPs
    during ripple events.

    Parameters
    ----------
    time : array_like, shape (n_time,)
    filtered_lfps : array_like, shape (n_time, n_signals)
        Bandpass filtered time series of electric potentials in the ripple band
    sampling_frequency : float
        Number of samples per second.
    ripple_times: DataFrame, shape (n_ripples, )
        Ripple start and end times, as returned by a detector.

    Returns
    -------
    ripple_features : pandas DataFrame
        Contains the mean (across channels) RMS of the LFP during each ripple event.
        Contains the summed (across channels) RMS of the LFP during each ripple event.
        Contains the max (across channels) RMS of the LFP during each ripple event.
    """
    ripple_features = pd.DataFrame(
        columns=["mean_rms", "summed_rms", "max_rms"], index=ripple_times.index
    )
    filtered_lfps = pd.DataFrame(filtered_lfps, index=pd.Index(time, name="time"))

    for ripple in ripple_times.itertuples():
        ripple_lfps = filtered_lfps[ripple.start_time : ripple.end_time]
        ripple_lfps_rms = np.sqrt(np.mean(ripple_lfps ** 2))
        ripple_features.loc[ripple.Index] = [
            np.mean(ripple_lfps_rms),
            np.sum(ripple_lfps_rms),
            np.max(ripple_lfps_rms),
        ]

    return ripple_features


def get_envelope_features_Kay(
    time, filtered_lfps, sampling_frequency, ripple_times, smoothing_sigma=0.004
):
    """Get features for ripples detected using the Kay method,
    where several channels are summed and their combined envelope
    is used for ripple detection. All parameters should be as passed
    to the Kay ripple detector.

    Parameters
    ----------
    time : array_like, shape (n_time,)
    filtered_lfps : array_like, shape (n_time, n_signals)
        Bandpass filtered time series of electric potentials in the ripple band
    sampling_frequency : float
        Number of samples per second.
    ripple_times: DataFrame, shape (n_ripples, )
        Ripple start and end times, as returned by the Kay detector.
    smoothing_sigma : float, optional
        Amount to smooth the time series over time. The default is
        given assuming time is in units of seconds.

    Returns
    -------
    ripple_features : pandas DataFrame
        Contains the integral of the envelope of the combined, filtered lfp.
        Also contains the size of this envelope's peak.
    """
    ripple_features = pd.DataFrame(
        columns=["envelope_integral", "envelope_peak"], index=ripple_times.index
    )

    filtered_lfps = get_envelope(filtered_lfps)
    combined_filtered_lfps = np.sum(filtered_lfps ** 2, axis=1)
    combined_filtered_lfps = gaussian_smooth(
        combined_filtered_lfps, smoothing_sigma, sampling_frequency
    )
    combined_filtered_lfps = np.sqrt(combined_filtered_lfps)

    combined_filtered_lfps = pd.DataFrame(
        combined_filtered_lfps, index=pd.Index(time, name="time")
    )
    for ripple in ripple_times.itertuples():
        combined_ripple_envelope = combined_filtered_lfps[
            ripple.start_time : ripple.end_time
        ]
        ripple_features.loc[ripple.Index] = [
            np.sum(combined_ripple_envelope[0]),
            np.max(combined_ripple_envelope[0]),
        ]

    return ripple_features


def get_envelope_features_Karlsson(
    time, filtered_lfps, sampling_frequency, ripple_times, smoothing_sigma=0.004
):
    """Get features for ripples detected using the Karlsson method,
    where envelopes are calculated separate for each channel and then
    used for ripple detection. All parameters should be as passed
    to the Karlsson ripple detector.

    Parameters
    ----------
    time : array_like, shape (n_time,)
    filtered_lfps : array_like, shape (n_time, n_signals)
        Bandpass filtered time series of electric potentials in the ripple band
    sampling_frequency : float
        Number of samples per second.
    ripple_times: DataFrame, shape (n_ripples, )
        Ripple start and end times, as returned by the Karlsson detector.
    smoothing_sigma : float, optional
        Amount to smooth the time series over time. The default is
        given assuming time is in units of seconds.

    Returns
    -------
    ripple_features : pandas DataFrame
        Contains the mean (across channels) integral of the envelope of the filtered lfp.
        Contains the summed (across channels) integrals of the envelope of the filtered lfp.
        Contains the max (across channels) peak of the envelope of the filtered lfp.
    """
    ripple_features = pd.DataFrame(
        columns=[
            "mean_envelope_integral",
            "summed_envelope_integrals",
            "max_envelope_peak",
        ],
        index=ripple_times.index,
    )

    filtered_lfps = get_envelope(filtered_lfps)
    filtered_lfps = gaussian_smooth(
        filtered_lfps, sigma=smoothing_sigma, sampling_frequency=sampling_frequency
    )

    filtered_lfps = pd.DataFrame(filtered_lfps, index=pd.Index(time, name="time"))
    for ripple in ripple_times.itertuples():
        ripple_envelopes = np.asarray(
            filtered_lfps[ripple.start_time : ripple.end_time]
        )
        ripple_features.loc[ripple.Index] = [
            np.mean(np.sum(ripple_envelopes, axis=0)),
            np.sum(ripple_envelopes),
            np.max(ripple_envelopes),
        ]

    return ripple_features


def get_ripple_amplitudes(time, filtered_lfps, ripple_times):
    """Compute and add 'mean_amplitude' and 'max_amplitude' fields to a ripple features
    dataframe.

    Parameters
    ----------
    time: array_like, shape (n_time,)
    filtered_lfps : array_like, shape (n_time, n_signals)
        Bandpass filtered time series of electric potentials in the ripple band
    ripple_times: DataFrame, shape (n_ripples, )
        Ripple start and end times, as returned by the detector.

    Returns
    -------
    ripple_features: DataFrame, shape (n_ripples, )
        Ripple start and end times, annotated with computer properties.
    """

    ripple_features = pd.DataFrame(
        columns=["mean_amplitude", "max_amplitude"], index=ripple_times.index
    )

    filtered_lfps = pd.DataFrame(filtered_lfps, index=pd.Index(time, name="time"))
    for ripple in ripple_times.itertuples():
        ripple_lfps = filtered_lfps[ripple.start_time : ripple.end_time]
        ripple_lfps_amplitudes = np.max(ripple_lfps) - np.min(ripple_lfps)
        ripple_features.loc[ripple.Index] = [
            np.mean(ripple_lfps_amplitudes),
            np.max(ripple_lfps_amplitudes),
        ]

    return ripple_features


def get_nadir_time(time, filtered_lfps, ripple):
    """Find the time of a ripple's deepest trough.

    Parameters
    ----------
    time: array_like, shape (n_time,)
    filtered_lfps : array_like, shape (n_time, n_signals)
        Bandpass filtered time series of electric potentials in the ripple band
    ripple: Series
        A row from a ripple features DataFrame, with 'start_time' and 'end_time' fields.

    Returns
    -------
    nadir_time: float
        The time of the ripple's deepest trough.
    """
    ripple_samples = np.where(
        np.logical_and(time >= ripple.start_time, time <= ripple.end_time)
    )
    ripple_time = time[ripple_samples]
    ripple_lfps = filtered_lfps[ripple_samples]
    (t_idx, ch_idx) = np.unravel_index(
        np.argmin(ripple_lfps, axis=None), ripple_lfps.shape
    )
    nadir_time = ripple_time[t_idx]

    return nadir_time


def compute_ripple_features(
    time,
    filtered_lfps,
    ripple_times,
    sampling_frequency,
    method,
    smoothing_sigma=0.004,
):
    """Compute all relevant ripple features.

    Parameters
    ----------
    time : array_like, shape (n_time,)
    filtered_lfps : array_like, shape (n_time, n_signals)
        Bandpass filtered time series of electric potentials in the ripple band
    sampling_frequency : float
        Number of samples per second.
    ripple_times: DataFrame, shape (n_ripples, )
        Ripple start and end times, as returned by the Karlsson detector.
    method : str
        The method-family used for detection, either 'Kay' or 'Karlsson'.
    smoothing_sigma : float, optional, default: 0.004
        The smoothing value used for ripple detection. Needed in order to accurately re-compute properties of the envelope used for detection.

    Returns
    -------
    ripple_features : DataFrame, shape (n_ripples, )
        Ripple times with all computed features.
    """

    ripple_times["duration"] = ripple_times.apply(
        lambda x: x.end_time - x.start_time, axis=1
    )
    ripple_times["center_time"] = ripple_times.apply(
        lambda x: x.start_time + x.duration / 2, axis=1
    )
    ripple_times["nadir_time"] = ripple_times.apply(
        lambda x: get_nadir_time(time, filtered_lfps, x), axis=1
    )

    rms_features = get_ripple_rms_features(time, filtered_lfps, ripple_times)
    amplitude_features = get_ripple_amplitudes(time, filtered_lfps, ripple_times)

    if method == "Kay":
        envelope_features = get_envelope_features_Kay(
            time, filtered_lfps, sampling_frequency, ripple_times
        )
    elif method == "Karlsson":
        envelope_features = get_envelope_features_Karlsson(
            time,
            filtered_lfps,
            sampling_frequency,
            ripple_times,
            smoothing_sigma=smoothing_sigma,
        )

    ripple_features = pd.concat(
        [ripple_times, envelope_features, rms_features, amplitude_features], axis=1
    )

    return ripple_features


def get_epoched_ripple_density(
    recording_start_time, recording_end_time, ripple_times, epoch_length
):
    epoch_start_times = np.arange(
        recording_start_time, recording_end_time, epoch_length
    )
    epoch_end_times = epoch_start_times + epoch_length
    epoch_center_times = (epoch_start_times + epoch_end_times) / 2
    epochs = pd.DataFrame(
        data={
            "start_time": epoch_start_times,
            "end_time": epoch_end_times,
            "center_time": epoch_center_times,
        }
    )
    epochs["n_ripples"] = epochs.apply(
        lambda epoch: np.sum(
            np.logical_and(
                ripple_times >= epoch.start_time, ripple_times <= epoch.end_time
            )
        ),
        axis=1,
    )
    epochs["ripple_density"] = epochs.apply(
        lambda epoch: epoch.n_ripples / (epoch.end_time - epoch.start_time), axis=1
    )

    return epochs