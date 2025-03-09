import logging

import elephant.statistics
import neo
import numpy as np
import pandas as pd
import quantities
import xarray as xr
from elephant import kernels

import ecephys.hypnogram as hyp
import ecephys.xrsig as xrsig
from ecephys.units import dtypes

from . import common

logger = logging.getLogger(__name__)


def get_mu_spindle_detection_params() -> dict:
    return dict(
        instantaneous_rate_sfreq_hz=256,  # Sampling frequency of gaussian-smoothed spike train
        instantaneous_rate_gaussian_sigma_msec=5,  # Width in msec of gaussian smoothing kernel
        sigma_lo=9,  # Minimum spindle frequency
        sigma_hi=16,  # Maximum spindle frequency
        broadband_lo=0.5,  # Minimum broadband frequency
        broadband_hi=None,  # Maximum broadband frequency
        sigma_filter_kwargs=dict(
            l_trans_bandwidth=1.5, h_trans_bandwidth=1.5, method="fir"
        ),  # Parameters for mne.filter.filter_data for sigma data
        stft_window=2.0,  # Relative power STFT
        stft_step=0.1,  # Relative power STFT
        rpow_convolution_window=0.3,  # Smoothing of sigma relative power STFT
        rpow_threshold=0.15,  # Smoothing of sigma relative power STFT
        mrms_window=0.5,  # Moving sigma RMS
        mrms_step=0.1,  # Moving sigma RMS
        mrms_stds_threshold=1.5,  # Threshold for sigma RMS power, in STDs of the estimation interval mean
        # mptp_window=0.3,  # Moving peak to peak
        # mptp_step=0.1,  # Moving  peak to peak
        # mptp_stds_threshold=1.5,  # Threshold for moving peak to peak
        decision_function_convolution_window=0.1,  # Improve spindle start/end time estimation by smoothing decision index by this much
        decision_function_threshold=1.001,  # Decision index above which an event is considered a spindle
        min_distance=0.3,  # Spindles closer than this will be merged together
        min_duration=0.5,  # Minimum spindle duration
        max_duration=2.5,  # Maximum spindle duration
    )


def _get_mu_spindle_properties(
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
    spiketrain_sec: dtypes.SpikeTrain_Secs,
    params: dict,
    hg: hyp.FloatHypnogram,
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

    mu = _compute_instantaneous_rate_xrsig(
        spiketrain_sec,
        params["instantaneous_rate_sfreq_hz"],
        params["instantaneous_rate_gaussian_sigma_msec"],
        t_start_sec=hg.start_time.min(),
        t_stop_sec=hg.end_time.max(),
        channel_name="mua",
    )
    mu_sigma = xrsig.mne_filter(
        mu,
        params["sigma_lo"],
        params["sigma_hi"],
        **params["sigma_filter_kwargs"],
        verbose=False,
    ).rename("Sigma-filtered instantaneous_rate")

    # Smoothed relative power
    spg = xrsig.stft_psd(
        mu,
        n_fft=int(mu.fs * params["stft_window"]),
        hop_len=int(mu.fs * params["stft_step"]),
    )
    rpow = common.get_relative_sigma_power(
        spg,
        params["sigma_lo"],
        params["sigma_hi"],
        params["broadband_lo"],
        params["broadband_hi"],
        mu.time.values,
    ).rename("Relative Sigma Power")
    # smooth
    w = int(params["rpow_convolution_window"] * mu.fs)
    rpow.data[:, 0] = np.convolve(rpow.data[:, 0], np.ones((w,)), mode="same") / w
    rpow = rpow.rename("Smoothed relative sigma power")
    # rpow_thresh = common.get_xrsig_thresholds(rpow, params["rpow_stds_threshold"], artifacts, hg, reference_state="NREM")
    rpow_thresh = params["rpow_threshold"]

    mrms = common.get_single_channel_moving_transform(
        mu_sigma, "rms", params["mrms_window"], params["mrms_step"]
    ).rename("Sigma RMS")
    mrms_thresh = common.get_xrsig_thresholds(
        mrms, params["mrms_stds_threshold"], artifacts, hg, reference_state="NREM"
    )

    # mptp = common.get_single_channel_moving_transform(mu, "ptp", params["mptp_window"], params["mptp_step"]).rename("Moving peak-to-peak")
    # mptp_thresh = common.get_xrsig_thresholds(mptp, params["mptp_stds_threshold"], artifacts, hg, reference_state="NREM")

    decision_function = common.get_decision_function(
        [
            (rpow, rpow_thresh),
            (mrms, mrms_thresh),
            # (mptp, mptp_thresh),
        ],
        params["decision_function_convolution_window"],
        mu.fs,
    ).rename("Decision Function")

    spindles, troughs = _get_mu_spindle_properties(
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
        rpow,
        rpow_thresh,
        mrms,
        mrms_thresh,
        # mptp,
        # mptp_thresh,
        decision_function,
        spindles,
        troughs,
    )


def examine_mu_spindle(
    spindles: pd.DataFrame,
    mu: xr.DataArray,
    mu_sigma: xr.DataArray,
    rpow: xr.DataArray,
    mrms: xr.DataArray,
    # mptp: xr.DataArray,
    decision_function: xr.DataArray,
    rpow_thresh: xr.DataArray,
    mrms_thresh: xr.DataArray,
    # mptp_thresh: xr.DataArray,
    decision_thresh: float,
    troughs=None,
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
            (rpow, rpow_thresh),
            (mrms, mrms_thresh),
            # (mptp, mptp_thresh),
            (decision_function, decision_thresh),
        ],
        troughs=troughs,
        plot_duration=plot_duration,
        i=i,
        t=t,
        channel=channel,
        hg=hg,
    )


def _convert_to_neo_spiketrain(
    spike_train_sec: dtypes.SpikeTrain_Secs,
    t_start_sec=None,
    t_stop_sec=None,
):
    if t_start_sec is None:
        t_start_sec = np.min(spike_train_sec)
    if t_stop_sec is None:
        t_stop_sec = np.max(spike_train_sec)

    sec = quantities.s
    return neo.SpikeTrain(
        spike_train_sec, t_start=t_start_sec * sec, t_stop=t_stop_sec * sec, units=sec
    )


def _compute_instantaneous_rate_xrsig(
    spiketrain_sec: dtypes.SpikeTrain_Secs,
    sampling_frequency_hz: float = 300,
    gaussian_sigma_msec: float = None,
    t_start_sec: float = None,
    t_stop_sec: float = None,
    channel_name: str = "mua",
) -> xr.DataArray:
    """Elephant-style instantaneous rate with gaussian kernel."""
    if gaussian_sigma_msec is None:
        kernel = "auto"
    else:
        kernel = kernels.GaussianKernel(
            sigma=gaussian_sigma_msec * quantities.ms,
        )

    neotrain = _convert_to_neo_spiketrain(
        spiketrain_sec,
        t_start_sec=min(t_start_sec, np.min(spiketrain_sec)),
        t_stop_sec=max(t_stop_sec, np.max(spiketrain_sec)),
    )

    res = elephant.statistics.instantaneous_rate(
        neotrain, (1 / sampling_frequency_hz) * quantities.s, kernel=kernel
    )

    return xr.DataArray(
        res.magnitude,
        dims=["time", "channel"],
        coords={
            "channel": [channel_name],
            "time": res.times.magnitude,
        },
        name="Instantaneous rate (Hz)",
        attrs={"fs": sampling_frequency_hz},
    )
