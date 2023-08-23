import ecephys.spindles.common as common
import logging
from ecephys.units.dtypes import SpikeTrain_Secs
from ecephys.units import elephantutils

from ecephys import hypnogram
from ecephys import xrsig
import numpy as np
import pandas as pd
import xarray as xr


logger = logging.getLogger(__name__)


def get_mu_spindle_detection_params() -> dict:
    return dict(
        instantaneous_rate_sfreq_hz=256, # Sampling frequency of gaussian-smoothed spike train
        instantaneous_rate_gaussian_sigma_msec=10, # Width in msec of gaussian smoothing kernel
        sigma_lo=9,  # Minimum spindle frequency
        sigma_hi=16,  # Maximum spindle frequency
        sigma_filter_kwargs=dict(
            l_trans_bandwidth=1.5, h_trans_bandwidth=1.5, method="fir"
        ),  # Parameters for mne.filter.filter_data for sigma data
        mrms_window=0.5,  # Moving sigma RMS
        mrms_step=0.1,  # Moving sigma RMS
        mrms_stds_threshold=1.5,  # Threshold for sigma RMS power, in STDs of the estimation interval mean
        decision_function_convolution_window=0.1,  # Improve spindle start/end time estimation by smoothing decision index by this much
        decision_function_threshold=0,  # Decision index above which an event is considered a spindle
        min_distance=0.3,  # Spindles closer than this will be merged together
        min_duration=0.5,  # Minimum spindle duration
        max_duration=2.5,  # Maximum spindle duration
    )


def get_mu_decision_function(
    mrms: xr.DataArray,
    mrms_threshold: float,
    convolution_window_length_sec: float,
    fs: float,
) -> xr.DataArray:
    idx_mrms = (mrms > mrms_threshold).astype(int)
    idx_sum = idx_mrms

    w = int(convolution_window_length_sec * fs)
    idx_sum.data[:, 0] = np.convolve(idx_sum.data[:, 0], np.ones((w,)), mode="same") / w

    return idx_sum


def get_mu_spindle_properties(
    mu: xr.DataArray,
    mu_sigma: xr.DataArray,
    decision_function: xr.DataArray,
    decision_threshold: float,
    min_distance: float,
    min_duration: float,
    max_duration: float,
    sigma_hi: float,
    hg=None,
):
    spindles, trough_times = common.get_base_spindle_properties(
        mu,
        mu_sigma,
        decision_function=decision_function,
        decision_threshold=decision_threshold,
        min_distance=min_distance,
        min_duration=min_duration,
        max_duration=max_duration,
        sigma_hi=sigma_hi,
        hg=hg,
    )

    # Add MU-specific info
    spindles["SpindleType"] = "mu"

    return spindles, trough_times


def detect_mu_spindles_from_spiketrain(
    spiketrain_sec: SpikeTrain_Secs,
    params: dict,
    hg: hypnogram.FloatHypnogram,
    artifacts: pd.DataFrame = None,
) -> tuple[
    xr.DataArray,
    xr.DataArray,
    xr.DataArray,
    xr.DataArray,
    xr.DataArray,
    xr.DataArray,
    xr.DataArray,
    pd.DataFrame,
    dict,
]:
    if artifacts is None:
        artifacts = pd.DataFrame()

    mu = elephantutils.compute_instantaneous_rate_xrsig(
        spiketrain_sec,
        params["instantaneous_rate_sfreq_hz"],
        params["instantaneous_rate_gaussian_sigma_msec"],
        t_start_sec = hg.start_time.min(),
        t_stop_sec = hg.end_time.max(),
        channel_name="mua",
    )
    mu_sigma = xrsig.mne_filter(
        mu, params["sigma_lo"], params["sigma_hi"], **params["sigma_filter_kwargs"], verbose=False,
    ).rename("Sigma-filtered instantaneous_rate")

    mrms = common.get_single_channel_mrms(mu_sigma, params["mrms_window"], params["mrms_step"]).rename("Sigma RMS")
    mrms_thresh = common.get_mrms_thresholds(mrms, params["mrms_stds_threshold"], artifacts, hg, reference_state="NREM")

    decision_function = get_mu_decision_function(
        mrms,
        mrms_thresh,
        params["decision_function_convolution_window"],
        mu.fs,
    ).rename("Decision Function")

    spindles, troughs = get_mu_spindle_properties(
        mu,
        mu_sigma,
        decision_function,
        params["decision_function_threshold"],
        params["min_distance"],
        params["min_duration"],
        params["max_duration"],
        params["sigma_hi"],
        hg,
    )

    return (
        mu_sigma,
        mu,
        mrms,
        mrms_thresh,
        decision_function,
        spindles,
        troughs,
    )


def examine_mu_spindle(
    spindles: pd.DataFrame,
    mu: xr.DataArray,
    mu_sigma: xr.DataArray,
    mrms: xr.DataArray,
    decision_function: xr.DataArray,
    mrms_thresh: xr.DataArray,
    decision_thresh: float,
    plot_duration: float = 6.0,
    i: int = None,
    t: float = None,
    channel: str = None,
    hg=None,
):
    common.examine_spindle(
        spindles,
        [
            (mu, None),
            (mu_sigma, None),
            (mrms, mrms_thresh),
            (decision_function, decision_thresh), 
        ],
        plot_duration=plot_duration,
        i=i,
        t=t,
        channel=channel,
        hg=hg,
    )