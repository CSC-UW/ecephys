import logging

from ecephys import hypnogram
from ecephys import plot as eplt
from ecephys import wne
from ecephys import xrsig
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal
import scipy.fftpack
import wisc_ecephys_tools as wet
import xarray as xr
import yasa


logger = logging.getLogger(__name__)


def get_spindle_detection_params() -> dict:
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


def get_relative_sigma_power(
    spgs: xr.DataArray,
    sigma_lo: float,
    sigma_hi: float,
    broadband_lo: float,
    broadband_hi: float,
    interpolation_times: np.ndarray = None,
) -> xr.DataArray:
    sigma = spgs.sel(frequency=slice(sigma_lo, sigma_hi)).sum(dim="frequency").T
    broad = spgs.sel(frequency=slice(broadband_lo, broadband_hi)).sum(dim="frequency").T
    rpow = sigma / broad
    if interpolation_times is not None:
        rpow = rpow.interp(
            time=interpolation_times,
            method="cubic",
            kwargs=dict(bounds_error=False, fill_value=0.0),
        )  # Interpolate results so that we have 1 estimate for each LFP sample
    return rpow


def select_best_spindle_channel(
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
    rpow_estm = get_relative_sigma_power(
        spgs_estm, sigma_lo, sigma_hi, broadband_lo, broadband_hi
    )
    rpow_profile = (
        rpow_estm.mean(dim="time").rolling(channel=smoothing, center=True).median()
    )
    best_channel = rpow_profile.channel[rpow_profile.argmax(dim="channel")]
    xrsig.plot_laminar_scalars_horizontal(rpow_profile, figsize=(32, 6))
    return best_channel


def get_single_channel_mrms(
    lf_sigma: xr.DataArray, window: float, step: float
) -> xr.DataArray:
    mrms = xr.zeros_like(lf_sigma)
    _, mrms.values[:, 0] = yasa.moving_transform(
        x=lf_sigma.values.squeeze(),
        sf=lf_sigma.fs,
        window=window,
        step=step,
        method="rms",
        interp=True,
    )
    return mrms


def get_single_channel_mcorr(
    lf_sigma: xr.DataArray, lf_broad: xr.DataArray, window: float, step: float
) -> xr.DataArray:
    mcorr = xr.zeros_like(lf_sigma)
    _, mcorr[:, 0] = yasa.moving_transform(
        x=lf_sigma.values.squeeze(),
        y=lf_broad.values.squeeze(),
        sf=lf_sigma.fs,
        window=window,
        step=step,
        method="corr",
        interp=True,
    )
    return mcorr


def get_spindle_properties(
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
    # We really don't need lf...
    assert lf.shape == lf_sigma.shape, "lf and lf_sigma must have the same shape"
    assert (
        lf.shape == decision_function.shape
    ), "lf and decision function must have the same shape"
    sf = lf.fs
    n_samples = lf.time.size
    nfast = scipy.fftpack.next_fast_len(n_samples)
    t0 = lf.time.values[0]
    min_distance_msec = min_distance * 1000

    # Hilbert power (to define the instantaneous frequency / power)
    analytic = scipy.signal.hilbert(lf_sigma.values.T, N=nfast)[:, :n_samples]
    inst_phase = np.angle(analytic)
    inst_pow = np.square(np.abs(analytic))
    inst_freq = sf / (2 * np.pi) * np.diff(inst_phase, axis=-1)

    # Initialize empty outputs
    df = pd.DataFrame()
    trough_times = dict()
    ch_names = lf.channel.values
    for i, ch in enumerate(ch_names):
        where_sp = np.where(
            decision_function.sel(channel=ch).values > (decision_threshold)
        )[0]

        # If no events are found, skip to next channel
        if not len(where_sp):
            logger.warning("No spindle were found in channel %s.", ch_names[i])
            continue

        # Merge events that are too close
        if min_distance_msec is not None and min_distance_msec > 0:
            where_sp = yasa.others._merge_close(where_sp, min_distance_msec, sf)

        # Extract start, end, and duration of each spindle
        sp = np.split(where_sp, np.where(np.diff(where_sp) != 1)[0] + 1)
        idx_start_end = lf.time.values[np.array([[k[0], k[-1]] for k in sp])]
        sp_start, sp_end = idx_start_end.T
        sp_dur = sp_end - sp_start

        # Find events with bad duration
        good_dur = np.logical_and(
            sp_dur > min_duration,
            sp_dur < max_duration,
        )

        # If no events of good duration are found, skip to next channel
        if all(~good_dur):
            logger.warning("No spindle were found in channel %s.", ch_names[i])
            continue

        # Initialize empty variables
        sp_amp = np.zeros(len(sp))
        sp_freq = np.zeros(len(sp))
        sp_rms = np.zeros(len(sp))
        sp_osc = np.zeros(len(sp))
        sp_sym = np.zeros(len(sp))
        sp_abs = np.zeros(len(sp))
        sp_rel = np.zeros(len(sp))
        sp_sta = np.zeros(len(sp))
        sp_pro = np.zeros(len(sp))
        trough_times[ch] = []

        # Number of oscillations (number of peaks separated by at least 60 ms)
        # --> 60 ms because 1000 ms / 16 Hz = 62.5 m, in other words, at 16 Hz,
        # peaks are separated by 62.5 ms. At 11 Hz peaks are separated by 90 ms
        min_peak_separation_msec = 1000 / (sigma_hi + 2)
        distance = min_peak_separation_msec * sf / 1000
        for j in np.arange(len(sp))[good_dur]:
            # Important: detrend the signal to avoid wrong PTP amplitude
            sp_x = np.arange(lf_sigma.values.T[i, sp[j]].size, dtype=np.float64)
            sp_det = yasa.numba._detrend(
                sp_x, lf_sigma.values.T[i, sp[j]].astype(np.float64)
            )
            sp_amp[j] = np.ptp(sp_det)  # Peak-to-peak amplitude
            sp_rms[j] = yasa.numba._rms(sp_det)  # Root mean square
            sp_rel[j] = np.median(
                rpow.sel(channel=ch).isel(time=sp[j]).values
            )  # Median relative power

            # Hilbert-based instantaneous properties
            sp_inst_freq = inst_freq[i, sp[j]]
            sp_inst_pow = inst_pow[i, sp[j]]
            sp_abs[j] = np.median(np.log10(sp_inst_pow[sp_inst_pow > 0]))
            sp_freq[j] = np.median(sp_inst_freq[sp_inst_freq > 0])

            # Number of oscillations
            peaks, peaks_params = scipy.signal.find_peaks(
                sp_det, distance=distance, prominence=(None, None)
            )
            sp_osc[j] = len(peaks)

            # Peak location & symmetry index
            # pk is expressed in sample since the beginning of the spindle
            pk = peaks[peaks_params["prominences"].argmax()]
            sp_pro[j] = lf.time.values[sp[j][0] + pk]
            sp_sym[j] = pk / sp_det.size

            troughs, trough_params = scipy.signal.find_peaks(
                -sp_det, distance=distance, prominence=(None, None)
            )
            troughs = troughs + sp[j][0]
            trough_times[ch].append(lf.time.values[troughs])

        # Create a dataframe
        sp_params = {
            "Start": sp_start,
            "Peak": sp_pro,
            "End": sp_end,
            "Duration": sp_dur,
            "Amplitude": sp_amp,
            "RMS": sp_rms,
            "AbsPower": sp_abs,
            "RelPower": sp_rel,
            "Frequency": sp_freq,
            "Oscillations": sp_osc,
            "Symmetry": sp_sym,
            "Stage": sp_sta,
        }

        df_chan = pd.DataFrame(sp_params)[good_dur]
        # ####################################################################
        # END SINGLE CHANNEL DETECTION
        # ####################################################################
        df_chan["Channel"] = ch
        df = pd.concat([df, df_chan], axis=0, ignore_index=True)

    if not len(df):
        return
    df[["Start", "Peak", "End"]] = df[["Start", "Peak", "End"]]
    # df["y"] = lf.y.sel(channel=df["Channel"].values)
    if hg is not None:
        df["Stage"] = hg.get_states(df["Start"])
    return df, trough_times


def get_mrms_thresholds(
    mrms: xr.DataArray,
    std_dev_threshold: float,
    artifacts: pd.DataFrame,
    hg: hypnogram.FloatHypnogram,
) -> xr.DataArray:
    t = mrms.time.values
    good_nrem = hg.keep_states(["NREM"]).covers_time(t)
    for artifact in artifacts.itertuples():
        times_in_bout = (t >= artifact.start_time) & (t <= artifact.end_time)
        good_nrem[times_in_bout] = False
    mrms_thresh = (
        mrms.isel(time=good_nrem).mean(dim="time")
        + mrms.isel(time=good_nrem).std(dim="time") * std_dev_threshold
    )
    return mrms_thresh


def get_decision_function(
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
    weight = xr.DataArray(np.ones(w) / w, dims=["convolution_window"])
    idx_sum = (
        idx_sum.rolling(time=w, center=True)
        .construct(time="convolution_window")
        .dot(weight)
    )
    idx_sum = idx_sum.fillna(0)

    return idx_sum


def detect_single_channel_spindles(
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
    rpow = get_relative_sigma_power(
        spg,
        params["sigma_lo"],
        params["sigma_hi"],
        params["broadband_lo"],
        params["broadband_hi"],
        lf_chan.time.values,
    ).rename("Relative Sigma Power")
    mrms = get_single_channel_mrms(
        lf_sigma, params["mrms_window"], params["mrms_step"]
    ).rename("Sigma RMS")
    mrms_thresh = get_mrms_thresholds(
        mrms, params["mrms_stds_threshold"], artifacts, hg
    )
    mcorr = get_single_channel_mcorr(
        lf_sigma, lf_broad, params["mcorr_window"], params["mcorr_step"]
    ).rename("Sigma-Broadband LFP Correlation")

    decision_function = get_decision_function(
        rpow,
        params["relative_sigma_power_threshold"],
        mcorr,
        params["mcorr_threshold"],
        mrms,
        mrms_thresh,
        params["decision_function_convolution_window"],
        lf_chan.fs,
    ).rename("Decision Function")
    spindles, troughs = get_spindle_properties(
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


def plot_spindle_timehist_summary(
    experiment: str, sglx_subject: wne.sglx.SGLXSubject, spindles: pd.DataFrame
) -> plt.Figure:
    s3 = sh.get_project("shared")
    delta = xr.open_dataarray(
        sh.get_cortical_bandpower_file(sglx_subject.name, experiment, "delta")
    )
    light_dark_periods, light_dark_period_labels = wet.shared.get_light_dark_periods(
        experiment, sglx_subject
    )
    hg = s3.load_float_hypnogram(experiment, sglx_subject.name)

    fig, axes = plt.subplots(4, 1, figsize=(32, 12))
    sh.plot_swa_timetrace(delta, axes[0], smoothing=8)
    sh.plot_event_rate_timetrace(spindles, axes[1], time_col="Start")
    sh.plot_event_amplitude_timehist(
        spindles, axes[2], time_col="Start", amp_col="Amplitude"
    )
    sh.plot_event_duration_timehist(
        spindles,
        axes[3],
        binwidth=(180, 0.2),
        time_col="Start",
        duration_col="Duration",
    )

    for ax in axes:
        eplt.plot_hypnogram_overlay(hg, ax=ax, state_colors=eplt.publication_colors)
        wet.shared.plot_lights_overlay(
            light_dark_periods, light_dark_period_labels, ax=ax, ymin=0, ymax=0.03
        )
        ax.set_xlim((light_dark_periods[0][0], light_dark_periods[-1][-1]))

    fig.suptitle(f"{sglx_subject.name}: {experiment}", fontsize=24)
    plt.tight_layout()
    return fig


def examine_spindle(
    spindles: pd.DataFrame,
    lf: xr.DataArray,
    lff_sigma: xr.DataArray,
    rpow: xr.DataArray,
    mrms: xr.DataArray,
    mcorr: xr.DataArray,
    decision_function: xr.DataArray,
    rpow_thresh: float,
    mrms_thresh: xr.DataArray,
    mcorr_thresh: float,
    decision_thresh: float,
    plot_duration: float = 6.0,
    i: int = None,
):
    if i is None:
        i = np.random.choice(len(spindles))
    evt = spindles.iloc[i]
    channel = evt.Channel

    t = evt.Start + evt.Duration / 2
    t1 = t - plot_duration / 2
    t2 = t + plot_duration / 2

    neighboring_evts = spindles.loc[(spindles["Start"] > t1) & (spindles["End"] < t2)]

    fig, axes = plt.subplots(6, 1, figsize=(16, 18), sharex=True)
    (ds_ax, sp_ax, rp_ax, rms_ax, corr_ax, idx_ax) = axes

    lf.sel(channel=channel, time=slice(t1, t2)).plot.line(x="time", ax=ds_ax)
    lff_sigma.sel(channel=channel, time=slice(t1, t2)).plot.line(x="time", ax=sp_ax)
    rpow.sel(channel=channel, time=slice(t1, t2)).plot.line(x="time", ax=rp_ax)
    rp_ax.axhline(rpow_thresh, c="r")
    mrms.sel(channel=channel, time=slice(t1, t2)).plot.line(x="time", ax=rms_ax)
    rms_ax.axhline(mrms_thresh.sel(channel=channel), c="r")
    mcorr.sel(channel=channel, time=slice(t1, t2)).plot.line(x="time", ax=corr_ax)
    corr_ax.axhline(mcorr_thresh, c="r")
    decision_function.sel(channel=channel, time=slice(t1, t2)).plot.line(
        x="time", ax=idx_ax
    )
    idx_ax.axhline(decision_thresh, c="r")

    for ax in axes:
        ax.axvline(evt.Start, c="g", ls=":")
        ax.axvline(evt.End, c="g", ls=":")

        for e in neighboring_evts.itertuples():
            ax.axvline(e.Start, c="g", ls=":")
            ax.axvline(e.End, c="g", ls=":")

        ax.set_title(None)

    return evt
