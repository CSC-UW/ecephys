import numpy as np
import pandas as pd
from ..utils import zscore_to_value
from .event_detection import threshold_by_value


def get_sink_amplitudes(events, time, sr_csd):
    mean_sr_csd = sr_csd.mean(axis=0)

    def _get_sink_amplitude(evt):
        evt_mask = np.logical_and(time >= evt.start_time, time <= evt.end_time)
        mean_evt_csd = mean_sr_csd[evt_mask]
        return np.min(mean_evt_csd)

    return events.apply(_get_sink_amplitude, axis=1)


def get_sink_integrals(events, time, fs, sr_csd):
    normed_sr_csd = sr_csd.mean(axis=0) / fs

    def _get_sink_integral(evt):
        evt_mask = np.logical_and(time >= evt.start_time, time <= evt.end_time)
        return normed_sr_csd[evt_mask].sum()

    return events.apply(_get_sink_integral, axis=1)


def detect_sharp_waves_by_zscore(
    time,
    sr_csd,
    detection_threshold_zscore=2.5,
    boundary_threshold_zscore=1,
    minimum_duration=0.005,
):
    """Find start and end times of sharp waves, done by thresholding the combined stratum radiatum CSD.

     Parameters
     ----------
     time : array_like, shape (n_time,)
     sr_csd: array_like, shape (n_estimates, n_time)
        Stratum radiatum CSD values. See notebook example.
     minimum_duration : float, optional
         Minimum time the z-score has to stay above threshold to be
         considered an event. The default is given assuming time is in
         units of seconds.
     detection_threshold_zscore : float, optional
         Number of standard deviations the combined CSD must exceed to
         be considered an event.
    boundary_threshold_zscore : float, optional
         Number of standard deviations the combined CSD must drop
         below to define the event start or end time.

     Returns
     -------
     spws : pandas DataFrame
    """

    combined_csd = np.sum(-sr_csd.T, axis=1)

    detection_threshold = zscore_to_value(combined_csd, detection_threshold_zscore)
    boundary_threshold = zscore_to_value(combined_csd, boundary_threshold_zscore)

    spws = detect_sharp_waves_by_value(
        time,
        sr_csd,
        detection_threshold=detection_threshold,
        boundary_threshold=boundary_threshold,
        minimum_duration=minimum_duration,
    )

    spws.attrs["detection_threshold_zscore"] = detection_threshold_zscore
    spws.attrs["boundary_threshold_zscore"] = boundary_threshold_zscore

    return spws


def detect_sharp_waves_by_value(
    time, sr_csd, detection_threshold, boundary_threshold, minimum_duration=0.005
):
    """Find start and end times of sharp waves, done by thresholding the combined stratum radiatum CSD.

     Parameters
     ----------
     time : array_like, shape (n_time,)
     sr_csd: array_like, shape (n_estimates, n_time)
        Stratum radiatum CSD values. See notebook example.
     minimum_duration : float, optional
         Minimum time the data has to stay above threshold to be
         considered an event. The default is given assuming time is in
         units of seconds.
     detection_threshold_zscore : float, optional
         Value the combined CSD must exceed to be considered an event.
    boundary_threshold_zscore : float, optional
         Value the combined CSD must drop below to define the event start or end time.

     Returns
     -------
     spws : pandas DataFrame
    """

    combined_csd = np.sum(-sr_csd.T, axis=1)

    candidate_spw_times = threshold_by_value(
        combined_csd,
        time,
        detection_threshold=detection_threshold,
        boundary_threshold=boundary_threshold,
        minimum_duration=minimum_duration,
    )

    index = pd.Index(np.arange(len(candidate_spw_times)) + 1, name="spw_number")
    spws = pd.DataFrame(
        candidate_spw_times, columns=["start_time", "end_time"], index=index
    )

    spws.attrs["detection_threshold"] = detection_threshold
    spws.attrs["boundary_threshold"] = boundary_threshold
    spws.attrs["minimum_duration"] = minimum_duration

    return spws
