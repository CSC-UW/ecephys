import logging
from ecephys import plot
import scipy.signal

from ecephys import hypnogram
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import yasa


logger = logging.getLogger(__name__)


def get_relative_sigma_power(
    spgs: xr.DataArray,
    sigma_lo: float,
    sigma_hi: float,
    broadband_lo: float,
    broadband_hi: float,
    interpolation_times: np.ndarray = None,
) -> xr.DataArray:
    # TODO: Maybe modify if npsig.stft is updated
    sigma = np.square(spgs.sel(frequency=slice(sigma_lo, sigma_hi))).sum(dim="frequency").T
    broad = np.square(spgs.sel(frequency=slice(broadband_lo, broadband_hi))).sum(dim="frequency").T
    rpow = sigma / broad
    if interpolation_times is not None:
        rpow = rpow.interp(
            time=interpolation_times,
            method="cubic",
            kwargs=dict(bounds_error=False, fill_value=0.0),
        )  # Interpolate results so that we have 1 estimate for each LFP sample
    return rpow


def get_single_channel_moving_transform(da_sigma: xr.DataArray, method: str, window: float, step: float) -> xr.DataArray:
    mda = xr.zeros_like(da_sigma)
    _, mda.values[:, 0] = yasa.moving_transform(
        x=da_sigma.values.squeeze(),
        sf=da_sigma.fs,
        window=window,
        step=step,
        method=method,
        interp=True,
    )
    return mda 


def get_single_channel_mcorr(
    da_sigma: xr.DataArray, da_broad: xr.DataArray, window: float, step: float
) -> xr.DataArray:
    mcorr = xr.zeros_like(da_sigma)
    _, mcorr[:, 0] = yasa.moving_transform(
        x=da_sigma.values.squeeze(),
        y=da_broad.values.squeeze(),
        sf=da_sigma.fs,
        window=window,
        step=step,
        method="corr",
        interp=True,
    )
    return mcorr


def get_decision_function(
    signal_threshold_tuples: list[tuple],
    convolution_window_length_sec: float,
    fs: float,
):
    idx_sum = np.sum(
        (da > thresh).astype(float) for da, thresh in signal_threshold_tuples
    )

    w = int(convolution_window_length_sec * fs)
    idx_sum.data[:, 0] = np.convolve(idx_sum.data[:, 0], np.ones((w,)), mode="same") / w

    return idx_sum

def get_base_spindle_properties(
    da: xr.DataArray,
    da_sigma: xr.DataArray,
    decision_function: xr.DataArray,
    decision_threshold: float,
    min_distance: float,
    min_duration: float,
    max_duration: float,
    sigma_hi: float,
    hg=None,
) -> pd.DataFrame:
    """Return spindles and troughs with properties."""
    # We really don't need da...
    assert da.shape == da_sigma.shape, "da and da_sigma must have the same shape"
    assert da.shape == decision_function.shape, "xrsig dataarray and decision function must have the same shape"
    sf = da.fs
    n_samples = da.time.size
    nfast = scipy.fftpack.next_fast_len(n_samples)
    t0 = da.time.values[0]
    min_distance_msec = min_distance * 1000

    # Hilbert power (to define the instantaneous frequency / power)
    analytic = scipy.signal.hilbert(da_sigma.values.T, N=nfast)[:, :n_samples]
    inst_phase = np.angle(analytic)
    inst_pow = np.square(np.abs(analytic))
    inst_freq = sf / (2 * np.pi) * np.diff(inst_phase, axis=-1)

    # Initialize empty outputs
    df = pd.DataFrame()
    trough_times = dict()
    ch_names = da.channel.values
    for i, ch in enumerate(ch_names):
        where_sp = np.where(decision_function.sel(channel=ch).values > (decision_threshold))[0]

        # If no events are found, skip to next channel
        if not len(where_sp):
            logger.warning("No spindle were found in channel %s.", ch_names[i])
            continue

        # Merge events that are too close
        if min_distance_msec is not None and min_distance_msec > 0:
            where_sp = yasa.others._merge_close(where_sp, min_distance_msec, sf)

        # Extract start, end, and duration of each spindle
        sp = np.split(where_sp, np.where(np.diff(where_sp) != 1)[0] + 1)
        idx_start_end = da.time.values[np.array([[k[0], k[-1]] for k in sp])]
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
            sp_x = np.arange(da_sigma.values.T[i, sp[j]].size, dtype=np.float64)
            sp_det = yasa.numba._detrend(sp_x, da_sigma.values.T[i, sp[j]].astype(np.float64))
            sp_amp[j] = np.ptp(sp_det)  # Peak-to-peak amplitude
            sp_rms[j] = yasa.numba._rms(sp_det)  # Root mean square

            # Hilbert-based instantaneous properties
            sp_inst_freq = inst_freq[i, sp[j]]
            sp_inst_pow = inst_pow[i, sp[j]]
            sp_abs[j] = np.median(np.log10(sp_inst_pow[sp_inst_pow > 0]))
            sp_freq[j] = np.median(sp_inst_freq[sp_inst_freq > 0])

            # Number of oscillations
            peaks, peaks_params = scipy.signal.find_peaks(sp_det, distance=distance, prominence=(None, None))
            sp_osc[j] = len(peaks)

            # Peak location & symmetry index
            # pk is expressed in sample since the beginning of the spindle
            pk = peaks[peaks_params["prominences"].argmax()]
            sp_pro[j] = da.time.values[sp[j][0] + pk]
            sp_sym[j] = pk / sp_det.size

            troughs, trough_params = scipy.signal.find_peaks(-sp_det, distance=distance, prominence=(None, None))
            troughs = troughs + sp[j][0]
            trough_times[ch].append(da.time.values[troughs])

        # Create a dataframe
        sp_params = {
            "Start": sp_start,
            "Peak": sp_pro,
            "End": sp_end,
            "Duration": sp_dur,
            "Amplitude": sp_amp,
            "RMS": sp_rms,
            "AbsPower": sp_abs,
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

    if hg is not None:
        df["Stage"] = hg.get_states(df["Start"])

    return df, trough_times


def get_xrsig_thresholds(
    da: xr.DataArray,
    std_dev_threshold: float,
    artifacts: pd.DataFrame,
    hg: hypnogram.FloatHypnogram,
    reference_state: str = "NREM",
) -> xr.DataArray:
    """Get threshold from distribution across reference state."""
    t = da.time.values
    good_nrem = hg.keep_states([reference_state]).covers_time(t)
    for artifact in artifacts.itertuples():
        times_in_bout = (t >= artifact.start_time) & (t <= artifact.end_time)
        good_nrem[times_in_bout] = False
    da_thresh = (
        da.isel(time=good_nrem).mean(dim="time") + da.isel(time=good_nrem).std(dim="time") * std_dev_threshold
    )
    return da_thresh 


def examine_spindle(
    spindles: pd.DataFrame,
    signal_threshold_tuples: list[tuple],
    troughs=None,
    plot_duration: float = 6.0,
    i: int = None,
    t: float = None,
    channel: str = None,
    hg = None,
):
    if t is None:
        if i is None:
            i = np.random.choice(len(spindles))
        evt = spindles.iloc[i]
        channel = evt.Channel

        t = evt.Start + evt.Duration / 2
    else:
        evt = None
        assert channel is not None
        assert i is None

    t1 = t - plot_duration / 2
    t2 = t + plot_duration / 2

    neighboring_evts = spindles.loc[(spindles["Start"] > t1) & (spindles["End"] < t2)]

    fig, axes = plt.subplots(len(signal_threshold_tuples), 1, figsize=(16, 8), sharex=True)

    for j, (sig, thresh) in enumerate(signal_threshold_tuples):
        ax = axes[j]
        sig.sel(channel=channel, time=slice(t1, t2)).plot.line(x="time", ax=ax)
        if isinstance(thresh, (int, float)):
            ax.axhline(thresh, c="r")
        if isinstance(thresh, (xr.DataArray)):
            ax.axhline(thresh.sel(channel=channel), c="r")

    for ax in axes:

        if evt is not None:
            ax.axvline(evt.Start, c="g", ls=":")
            ax.axvline(evt.End, c="g", ls=":")

            if troughs is not None:
                for trough_t in troughs[evt["Channel"]][i]:
                    ax.axvline(trough_t, c="b", ls=":", lw=0.5)

        for e in neighboring_evts.itertuples():
            ax.axvline(e.Start, c="g", ls=":")
            ax.axvline(e.End, c="g", ls=":")

        ax.set_title(None)

        if hg is not None:
            plot.plot_hypnogram_overlay(hg, ax=ax)
    
    if evt is not None:
        items = [f"{k}: {v}" for k, v in evt.items()]
        items.insert(5, "\n")
        title = ", ".join(items)
        fig.suptitle(title)
        return evt
