import pandas as pd
import numpy as np
from . import event_detection as evt
from ..utils import dt_series_to_seconds, round_to_values, all_arrays_equal, get_epocs
from statsmodels.nonparametric.smoothers_lowess import lowess

# -------------------- Plotting related imports --------------------
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from ipywidgets import (
    BoundedIntText,
    Checkbox,
    FloatSlider,
    HBox,
    SelectionSlider,
    fixed,
    interactive_output,
)
from neurodsp.plts.utils import check_ax
from ..plot import _lfp_explorer, _colormesh_timeseries_explorer
from sglxarray import load_trigger

# -------------------- Detection related functions --------------------


def get_detection_series(
    csd, coarse_detection_chans=slice(None), n_fine_detection_chans=5
):
    _csd = (
        csd.sel(channel=coarse_detection_chans)
        .rolling(channel=n_fine_detection_chans, center=True)
        .mean()
        .dropna(dim="channel")
    )
    return -_csd[_csd.argmin(dim="channel")]


def get_peak_info(sig, spws):
    spws = spws.copy()

    def _get_peak_info(spw):
        spw_sig = sig.sel(time=slice(spw.start_time, spw.end_time))
        peak = spw_sig[spw_sig.argmax()]
        return peak.item(), peak.time.item(), peak.channel.item()

    info = list(map(_get_peak_info, spws.itertuples()))
    spws[["peak_amplitude", "peak_time", "peak_channel"]] = info
    return spws


def get_coarse_detection_chans(peak_channel, n_coarse_detection_chans, csd_chans):
    assert (
        n_coarse_detection_chans % 2
    ), "Must use an odd number of of detection channels."

    idx = csd_chans.index(peak_channel)
    first = idx - n_coarse_detection_chans // 2
    last = idx + n_coarse_detection_chans // 2 + 1

    assert first >= 0, "Cannot detect SPWs outside the bounds of your CSD."
    assert last < len(csd_chans), "Cannot detect SPWs outside the bounds of your CSD."

    return csd_chans[first:last]


def detect_by_value(
    csd,
    initial_peak_channel,
    n_coarse_detection_chans,
    detection_threshold,
    boundary_threshold,
    minimum_duration=0.005,
):
    csd = csd.swap_dims({"pos": "channel"})
    cdc = get_coarse_detection_chans(
        initial_peak_channel, n_coarse_detection_chans, csd.channel.values.tolist()
    )
    ser = get_detection_series(csd, cdc)

    spws = evt.detect_by_value(
        ser.values,
        ser.time.values,
        detection_threshold,
        boundary_threshold,
        minimum_duration,
    )
    spws.attrs["initial_peak_channel"] = initial_peak_channel
    spws.attrs["n_coarse_detection_chans"] = n_coarse_detection_chans
    if "datetime" in ser.coords:
        spws.attrs["t0"] = ser.datetime.values.min()
    return get_peak_info(ser, spws)


def detect_by_zscore(
    csd,
    initial_peak_channel,
    n_coarse_detection_chans,
    detection_threshold=2.5,
    boundary_threshold=1.5,
    minimum_duration=0.005,
):
    csd = csd.swap_dims({"pos": "channel"})
    cdc = get_coarse_detection_chans(
        initial_peak_channel, n_coarse_detection_chans, csd.channel.values.tolist()
    )
    ser = get_detection_series(csd, cdc)

    spws = evt.detect_by_zscore(
        ser.values,
        ser.time.values,
        detection_threshold,
        boundary_threshold,
        minimum_duration,
    )
    spws.attrs["initial_peak_channel"] = initial_peak_channel
    spws.attrs["n_coarse_detection_chans"] = n_coarse_detection_chans
    if "datetime" in ser.coords:
        spws.attrs["t0"] = ser.datetime.values.min()
    return get_peak_info(ser, spws)


# -------------------- Drift related functions --------------------


def _estimate_drift(t, pos, **kwargs):
    out = lowess(pos, t, **kwargs)
    return out[:, 0], out[:, 1]


def estimate_drift(spws, imec_map, frac=1 / 48, it=3):
    um_per_mm = 1000
    peak_times = dt_series_to_seconds(spws.peak_time)
    peak_ycoords = imec_map.chans2coords(spws.peak_channel)[:, 1] / um_per_mm
    t, y = _estimate_drift(peak_times, peak_ycoords, frac=frac, it=it)
    nearest_y = round_to_values(y, imec_map.y / um_per_mm)
    t0_chan = imec_map.lf_map.set_index("y").loc[nearest_y[0] * um_per_mm]
    nearest_chans = imec_map.lf_map.set_index("y").loc[nearest_y * um_per_mm]
    nearest_ids = nearest_chans.chan_id.values
    usr_order_shift = nearest_chans.usr_order.values - t0_chan.usr_order

    return pd.DataFrame(
        {
            "dt": spws.peak_time.values,
            "t": t,
            "y": y,
            "nearest_y": nearest_y,
            "nearest_id": nearest_ids,
            "shifts": usr_order_shift,
        }
    )


def get_drift_epocs(drift):
    cols = ["nearest_y", "nearest_id", "shifts"]
    dfs = list(get_epocs(drift, col, "dt") for col in cols)
    assert all_arrays_equal(
        df.index for df in dfs
    ), "Different columns yielded different epochs."

    return pd.concat(dfs, axis=1)


def get_shift_timeseries(epocs, times):
    shifts = np.full(len(times), np.nan)
    for epoc in epocs.reset_index().itertuples():
        times_in_epoc = (times >= epoc.start_dt) & (times <= epoc.end_dt)
        shifts[times_in_epoc] = epoc.shifts

    mask = np.isnan(shifts)
    shifts[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), shifts[~mask])

    return shifts.astype(np.int64)


# -------------------- Plotting functions --------------------


def _spw_explorer(
    time,
    hpc_lfps,
    hpc_csd,
    spws,
    axes,
    window_length=1,
    spw_number=1,
):

    for ax in axes:
        ax.cla()

    # Compute peri-event window
    spw = spws.loc[spw_number]
    window_start_time = spw.midpoint - window_length / 2
    window_end_time = spw.midpoint + window_length / 2

    window_mask = np.logical_and(time >= window_start_time, time <= window_end_time)
    hpc_lfps = hpc_lfps[window_mask]
    hpc_csd = hpc_csd[window_mask]
    time = time[window_mask]

    # Plot raw LFPs
    _lfp_explorer(time, hpc_lfps, axes[0])

    # Plot CSD
    _colormesh_timeseries_explorer(time, hpc_csd.T, ax=axes[1])
    axes[1].set_xlim(axes[0].get_xlim())
    axes[1].set_xlabel("Time [sec]")
    axes[1].set_ylabel("Depth (mm)")

    # Highlight each ripple
    for spw in spws.itertuples():
        if (spw.start_time >= window_start_time) and (spw.end_time <= window_end_time):
            axes[0].axvspan(
                spw.start_time, spw.end_time, alpha=0.3, color="red", zorder=1000
            )
            axes[1].axvspan(
                spw.start_time, spw.end_time, alpha=0.3, color="red", zorder=1000
            )


def _lazy_spw_explorer(
    spws,
    metadata,
    subject,
    condition,
    axes,
    window_length=1,
    spw_number=1,
    n_plot_chans=None,
    i_chan=0,
    vspace=300,
    show_lfps=True,
    show_csd=True,
):
    lfp_ax, csd_ax = axes
    lfp_ax.cla()
    csd_ax.cla()

    # Compute peri-event window
    spw = spws.loc[spw_number]
    window_start_time = spw.midpoint - window_length / 2
    window_end_time = spw.midpoint + window_length / 2

    # Load the peri-event data
    all_chans = channel_groups.full[subject]
    (time, lfps, fs) = load_trigger(
        paths.get_datapath(subject=subject, condition=condition, data="lf.bin"),
        all_chans,
        start_time=window_start_time,
        end_time=window_end_time,
    )

    # Select subset of data used for detection
    idx_detection_chans = np.isin(all_chans, metadata["csd_chans"])
    lfps = lfps[:, idx_detection_chans]
    chan_labels = all_chans[idx_detection_chans]

    # Select further subset of data selected for plotting
    n_data_chans = lfps.shape[1]
    n_plot_chans = n_data_chans if n_plot_chans is None else n_plot_chans
    if (i_chan + n_plot_chans) > n_data_chans:
        i_chan = n_data_chans - n_plot_chans

    lfps = lfps[:, i_chan : i_chan + n_plot_chans]
    chan_labels = chan_labels[i_chan : i_chan + n_plot_chans]

    # Plot raw LFPs
    lfp_ax.set_facecolor("none")
    if show_lfps:
        _lfp_explorer(
            time, lfps, lfp_ax, chan_labels=chan_labels, vspace=vspace, zero_mean=True
        )
        lfp_ax.margins(y=0)

    # Compute CSD
    if show_csd:
        k = get_kcsd(
            lfps,
            do_lcurve=False,
            intersite_distance=metadata["intersite_distance"],
            gdx=metadata["gdx"],
            lambd=metadata["lambd"],
            R_init=metadata["R"],
        )
        csd = k.values("CSD")

    # Plot CSD
    csd_ax.set_zorder(lfp_ax.get_zorder() - 1)
    csd_ax.set_xlabel("Time [sec]")
    if show_csd:
        _colormesh_timeseries_explorer(time, csd.T, csd_ax, y=k.estm_x, flip_dv=True)
        csd_ax.set_ylabel("Depth (mm)")

    # Plot location of detection channels on CSD axes
    show_detection_chans = False
    if show_detection_chans:
        sr_chan_mask = np.isin(metadata["csd_chans"], metadata["detection_chans"])
        csd_ax.axhline(
            np.max(metadata["electrode_positions"][sr_chan_mask]),
            color="r",
            alpha=0.5,
            linestyle=":",
        )
        csd_ax.axhline(
            np.min(metadata["electrode_positions"][sr_chan_mask]),
            color="r",
            alpha=0.5,
            linestyle=":",
        )

    # Highlight each ripple
    for spw in spws.itertuples():
        if (spw.start_time >= window_start_time) and (spw.end_time <= window_end_time):
            lfp_ax.axvspan(
                spw.start_time, spw.end_time, alpha=0.1, color="red", zorder=1000
            )
            csd_ax.axvspan(
                spw.start_time, spw.end_time, fill=False, linestyle=":", zorder=1000
            )


def lazy_spw_explorer(spws, metadata, subject, condition, figsize=(20, 8)):
    """
    Examples
    --------
    %matplotlib widget
    lazy_spw_explorer(spws, metadata, subject, condition)
    """
    # Create interactive widgets for controlling plot parameters
    window_length = FloatSlider(
        min=0.25, max=4.0, step=0.25, value=1.0, description="Secs"
    )
    spw_number = SelectionSlider(
        options=spws.index.values, value=spws.index.values[0], description="SPW number"
    )
    # spw_number = BoundedIntText(
    #   min=1, max=spws.index.max(), step=1, value=1, description="SPW number"
    # )
    # _spw_number = BoundedIntText(
    #    min=1, max=spws.index.max(), step=1, value=1, description="SPW number"
    # )
    # jslink(
    #    (spw_number, "value"), (_spw_number, "value")
    # )  # Allow control from either widget for easy navigation
    n_plot_chans = BoundedIntText(min=1, max=1000, value=1000, description="nCh")
    i_chan = BoundedIntText(min=0, max=999, value=0, description="Ch")
    vspace = BoundedIntText(min=0, max=1000000, value=300, description="V Space")
    show_lfps = Checkbox(value=True, description="LFP")
    show_csd = Checkbox(value=True, description="CSD")

    # Lay control widgets out horizontally
    ui = HBox(
        [
            window_length,
            spw_number,
            #        _spw_number,
            n_plot_chans,
            i_chan,
            vspace,
            show_lfps,
            show_csd,
        ]
    )

    # Plot and display
    _, ax = plt.subplots(figsize=figsize)
    (lfp_ax, csd_ax) = (ax, ax.twinx())
    out = interactive_output(
        _lazy_spw_explorer,
        {
            "spws": fixed(spws),
            "metadata": fixed(metadata),
            "subject": fixed(subject),
            "condition": fixed(condition),
            "axes": fixed((lfp_ax, csd_ax)),
            "window_length": window_length,
            "spw_number": spw_number,
            "n_plot_chans": n_plot_chans,
            "i_chan": i_chan,
            "vspace": vspace,
            "show_lfps": show_lfps,
            "show_csd": show_csd,
        },
    )

    display(ui, out)


def plot_spw_density(spws, binwidth=10, ax=None, figsize=(20, 4)):
    ax = check_ax(ax, figsize=figsize)
    g = sns.histplot(
        spws.start_time, binwidth=binwidth, stat="frequency", color="black", ax=ax
    )

    seconds_per_hour = 3600
    hours = int(np.ceil(spws.end_time.max() / seconds_per_hour))
    xticks = [x * seconds_per_hour for x in range(hours)]
    xticklabels = [str(x) for x in range(hours)]
    g.set(
        xticks=xticks,
        xticklabels=xticklabels,
        xlabel=f"Hours from 9AM",
        ylabel="Density (events per second)",
        title=f"SPW Density ({binwidth}s bins)",
    )
    plt.show(g)
