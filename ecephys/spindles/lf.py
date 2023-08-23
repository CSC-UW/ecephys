import ecephys.spindles.common as common
import logging

from ecephys import hypnogram
from ecephys import xrsig
import numpy as np
import pandas as pd
import scipy.signal
import scipy.fftpack
import xarray as xr
import yasa


logger = logging.getLogger(__name__)


def get_lf_spindle_detection_params() -> dict:
    return dict(
        sigma_lo=9,  # Minimum spindle frequency
        sigma_hi=16,  # Maximum spindle frequency
        broadband_lo=0.5,  # Minimum broadband frequency
        broadband_hi=None,  # Maximum broadband frequency
        sigma_filter_kwargs=dict(
            l_trans_bandwidth=1.5, h_trans_bandwidth=1.5, method="fir"
        ),  # Parameters for mne.filter.filter_data for sigma data
        broadband_filter_kwargs=dict(method="fir"),
        stft_window=2.0,  # Relative power STFT
        stft_step=0.200,  # Relative power STFT
        mrms_window=0.3,  # Moving sigma RMS
        mrms_step=0.1,  # Moving sigma RMS
        mcorr_window=0.3,  # Moving sigma-broadband LFP correlation
        mcorr_step=0.1,  # Moving sigma-broadband LFP correlation
        relative_sigma_power_threshold=0.2,  # Threshold for relative sigma power
        mcorr_threshold=0.65,  # Threshold for sigma-broadband LFP correlation
        mrms_stds_threshold=1.5,  # Threshold for sigma RMS power, in STDs of the estimation interval mean
        decision_function_convolution_window=0.1,  # Improve spindle start/end time estimation by smoothing decision index by this much
        decision_function_threshold=2,  # Decision index above which an event is considered a spindle
        min_distance=0.3,  # Spindles closer than this will be merged together
        min_duration=0.5,  # Minimum spindle duration
        max_duration=2.5,  # Maximum spindle duration
    )


def select_best_lf_spindle_channel(
    lf: xr.DataArray,
    estm_start: float,
    estm_end: float,
    sigma_lo: float,
    sigma_hi: float,
    broadband_lo: float = 0.0,
    broadband_hi: float = np.Inf,
    smoothing: int = 5,
) -> xr.DataArray:
    lf_estm = lf.sel(time=slice(estm_start, estm_end))
    spgs_estm = xrsig.stft(lf_estm)
    rpow_estm = common.get_relative_sigma_power(spgs_estm, sigma_lo, sigma_hi, broadband_lo, broadband_hi)
    rpow_profile = rpow_estm.mean(dim="time").rolling(channel=smoothing, center=True).median()
    best_channel = rpow_profile.channel[rpow_profile.argmax(dim="channel")]
    xrsig.plot_laminar_scalars_horizontal(rpow_profile, figsize=(32, 6))
    return best_channel


def get_lf_decision_function(
    rpow: xr.DataArray,
    rpow_threshold: float,
    mcorr: xr.DataArray,
    mcorr_threshold: float,
    mrms: xr.DataArray,
    mrms_threshold: float,
    convolution_window_length_sec: float,
    fs: float,
) -> xr.DataArray:
    idx_rel_pow = (rpow > rpow_threshold).astype(int)
    idx_mcorr = (mcorr > mcorr_threshold).astype(int)
    idx_mrms = (mrms > mrms_threshold).astype(int)
    idx_sum = idx_rel_pow + idx_mcorr + idx_mrms

    w = int(convolution_window_length_sec * fs)
    idx_sum.data[:, 0] = np.convolve(idx_sum.data[:, 0], np.ones((w,)), mode="same") / w

    return idx_sum


def get_lf_spindle_properties(
    lf: xr.DataArray,
    lf_sigma: xr.DataArray,
    decision_function: xr.DataArray,
    rpow: xr.DataArray,
    decision_threshold: float,
    min_distance: float,
    min_duration: float,
    max_duration: float,
    sigma_hi: float,
    hg=None,
) -> pd.DataFrame:
    spindles, trough_times = common.get_base_spindle_properties(
        lf,
        lf_sigma,
        decision_function=decision_function,
        decision_threshold=decision_threshold,
        min_distance=min_distance,
        min_duration=min_duration,
        max_duration=max_duration,
        sigma_hi=sigma_hi,
        hg=hg,
    )

    # Add LF-specific fields
    spindles["SpindleType"] = "lf"
    spindles["y"] = lf.y.sel(channel=spindles["Channel"].values)
    # Relative power
    sp_rel = np.zeros(len(spindles))
    for i, row in spindles.itertuples():
        ch = row["Channel"]
        sp_rel[i] = np.median(
            rpow.sel(channel=ch).sel(time=slice(row["Start"], row["End"])).values
        )  # Median relative power

    return spindles, trough_times


def detect_single_channel_lf_spindles(
    lf_chan: xr.DataArray,
    params: dict,
    hg: hypnogram.FloatHypnogram,
    artifacts: pd.DataFrame,
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
    lf_chan.load()
    lf_sigma = xrsig.mne_filter(
        lf_chan, params["sigma_lo"], params["sigma_hi"], **params["sigma_filter_kwargs"]
    ).rename("Sigma LFP")
    lf_broad = xrsig.mne_filter(
        lf_chan,
        params["broadband_lo"],
        params["broadband_hi"],
        **params["broadband_filter_kwargs"],
    ).rename("Broadband LFP")
    spg = xrsig.stft(
        lf_chan,
        n_fft=int(lf_chan.fs * params["stft_window"]),
        hop_len=int(lf_chan.fs * params["stft_step"]),
    )
    rpow = common.get_relative_sigma_power(
        spg,
        params["sigma_lo"],
        params["sigma_hi"],
        params["broadband_lo"],
        params["broadband_hi"],
        lf_chan.time.values,
    ).rename("Relative Sigma Power")
    mrms = common.get_single_channel_mrms(lf_sigma, params["mrms_window"], params["mrms_step"]).rename("Sigma RMS")
    mrms_thresh = common.get_mrms_thresholds(mrms, params["mrms_stds_threshold"], artifacts, hg)
    mcorr = common.get_single_channel_mcorr(lf_sigma, lf_broad, params["mcorr_window"], params["mcorr_step"]).rename(
        "Sigma-Broadband LFP Correlation"
    )

    decision_function = get_lf_decision_function(
        rpow,
        params["relative_sigma_power_threshold"],
        mcorr,
        params["mcorr_threshold"],
        mrms,
        mrms_thresh,
        params["decision_function_convolution_window"],
        lf_chan.fs,
    ).rename("Decision Function")
    spindles, troughs = get_lf_spindle_properties(
        lf_chan,
        lf_sigma,
        decision_function,
        rpow,
        params["decision_function_threshold"],
        params["min_distance"],
        params["min_duration"],
        params["max_duration"],
        params["sigma_hi"],
        hg,
    )
    return (
        lf_sigma,
        lf_broad,
        rpow,
        mrms,
        mrms_thresh,
        mcorr,
        decision_function,
        spindles,
        troughs,
    )
