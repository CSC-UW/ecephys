import pandas as pd
import numpy as np
from scipy.stats import zscore
from ripple_detection.core import (
    _extend_segment,
    segment_boolean_series,
)
from ..utils import zscore_to_value


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


def threshold_by_value(
    data, time, detection_threshold, boundary_threshold, minimum_duration=0.015
):
    """Determine whether it exceeds the threshold values.

    Parameters
    ----------
    data : array_like, shape (n_time,)
    detection_threshold_zscore : int, optional
        The data must exceed this threshold in order for an event to be detected.
    boundary_threshold_zscore: int, optional
        Once an event is detected, the event's boundaries are defined as the time that the data drop below this threshold.
    minimum_duration: float, optional
        The minimum time that an event must persist.

    Returns
    -------
    candidate_event_times : pandas Dataframe
    """
    is_above_boundary_threshold = data >= boundary_threshold
    is_above_detection_threshold = data >= detection_threshold

    return extend_detection_threshold_to_boundaries(
        is_above_boundary_threshold,
        is_above_detection_threshold,
        time,
        minimum_duration=minimum_duration,
    )


def threshold_by_zscore(
    data,
    time,
    detection_threshold_zscore=2,
    boundary_threshold_zscore=0,
    minimum_duration=0.015,
):
    """Standardize the data and determine whether it exceeds the threshold values.

    Parameters
    ----------
    data : array_like, shape (n_time,)
    detection_threshold_zscore : int, optional
        The data must exceed this threshold in order for an event to be detected.
    boundary_threshold_zscore: int, optional
        Once an event is detected, the event's boundaries are defined as the time that the data drop below this threshold.
    minimum_duration: float, optional
        The minimum time that an event must persist.

    Returns
    -------
    candidate_event_times : pandas Dataframe
    """
    zscored_data = zscore(data)
    return threshold_by_value(
        zscored_data,
        time,
        detection_threshold_zscore,
        boundary_threshold_zscore,
        minimum_duration,
    )


def detect_by_value(
    data, time, detection_threshold, boundary_threshold, minimum_duration=0.005
):
    """Find start and end times of events, done by thresholding the detection signal.

    Parameters
    ----------
    data: array_like, shape (n_time,)
    time : array_like, shape (n_time,)
    minimum_duration : float, optional
        Minimum time the data has to stay above threshold to be
        considered an event. The default is given assuming time is in
        units of seconds.
    detection_threshold : float, optional
        Value the data must exceed to be considered an event.
    boundary_threshold : float, optional
        Value the data must drop below to define the event start or end time.

    Returns
    -------
    events : pandas DataFrame
    """

    candidate_event_times = threshold_by_value(
        data,
        time,
        detection_threshold=detection_threshold,
        boundary_threshold=boundary_threshold,
        minimum_duration=minimum_duration,
    )

    events = pd.DataFrame(candidate_event_times, columns=["start_time", "end_time"])

    events.attrs["detection_threshold"] = detection_threshold
    events.attrs["boundary_threshold"] = boundary_threshold
    events.attrs["minimum_duration"] = minimum_duration

    return events


def detect_by_zscore(
    data,
    time,
    detection_threshold_zscore=2.5,
    boundary_threshold_zscore=1,
    minimum_duration=0.005,
):
    """Find start and end times of events, done by thresholding the detection signal.

     Parameters
     ----------
    data: array_like, shape (n_time,)
    time : array_like, shape (n_time,)
    minimum_duration : float, optional
        Minimum time the data has to stay above threshold to be
        considered an event. The default is given assuming time is in
        units of seconds.
    detection_threshold_zscore : float, optional
        Number of standard deviations the data must exceed to
        be considered an event.
    boundary_threshold_zscore : float, optional
        Number of standard deviations the data must drop
        below to define the event start or end time.

    Returns
    -------
    events : pandas DataFrame
    """
    detection_threshold = zscore_to_value(data, detection_threshold_zscore)
    boundary_threshold = zscore_to_value(data, boundary_threshold_zscore)

    events = detect_by_value(
        data,
        time,
        detection_threshold=detection_threshold,
        boundary_threshold=boundary_threshold,
        minimum_duration=minimum_duration,
    )

    events.attrs["detection_threshold_zscore"] = detection_threshold_zscore
    events.attrs["boundary_threshold_zscore"] = boundary_threshold_zscore

    return events


# ========== Are any of the following functions still used? ==========


def get_events_in_interval(events, start_time, end_time, criteria=["midpoint"]):
    meets_criteria = np.full(len(events), True)
    for criterion in criteria:
        meets_criterion = np.logical_and(
            events[criterion] >= start_time, events[criterion] <= end_time
        )
        meets_criteria = np.logical_and(meets_criteria, meets_criterion)

    return events[meets_criteria]


def count_events_in_interval(events, start_time, end_time, criteria=["midpoint"]):
    return len(get_events_in_interval(events, start_time, end_time, criteria=criteria))


def get_epoched_event_density(
    events, first_epoch_start_time, last_epoch_end_time, epoch_length
):
    epochs = make_epochs(first_epoch_start_time, last_epoch_end_time, epoch_length)
    epochs["event_count"] = epochs.apply(
        lambda epoch: count_events_in_interval(
            events, epoch.start_time, epoch.end_time
        ),
        axis=1,
    )
    epochs["event_density"] = epochs.apply(
        lambda epoch: epoch.event_count / (epoch.end_time - epoch.start_time), axis=1
    )

    return epochs


def make_epochs(start_time, end_time, epoch_length):
    epoch_start_times = np.arange(start_time, end_time, epoch_length)
    epoch_end_times = epoch_start_times + epoch_length
    epoch_midpoints = (epoch_start_times + epoch_end_times) / 2
    epochs = pd.DataFrame(
        data={
            "start_time": epoch_start_times,
            "end_time": epoch_end_times,
            "midpoint": epoch_midpoints,
        }
    )

    return epochs


def get_durations(events):
    return events.apply(lambda evt: evt.end_time - evt.start_time, axis=1)


def get_midpoints(events):
    return events.apply(lambda evt: evt.start_time + evt.duration / 2, axis=1)


def add_states_to_events(events, hypnogram):
    events["state"] = None
    for index, bout in hypnogram.iterrows():
        events_in_bout = np.logical_and(
            events["start_time"] >= bout["start_time"],
            events["start_time"] < bout["end_time"],
        )
        events.loc[events_in_bout, "state"] = bout["state"]

    return events
