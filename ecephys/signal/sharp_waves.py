import pandas as pd
import numpy as np
from . import event_detection as evt
from ..utils import (
    dt_series_to_seconds,
    round_to_values,
    all_arrays_equal,
    get_epocs,
    load_df_h5,
)
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
from ..plot import lfp_explorer, colormesh_explorer
from sglxarray import load_trigger
from ..xrsig import get_kcsd


def load_spws(path, use_datetime=True):
    spws = load_df_h5(path)

    if use_datetime:
        t0 = pd.to_datetime(spws.attrs["t0"])
        spws["start_time"] = t0 + pd.to_timedelta(spws["start_time"], "s")
        spws["end_time"] = t0 + pd.to_timedelta(spws["end_time"], "s")
        spws["peak_time"] = t0 + pd.to_timedelta(spws["peak_time"], "s")

    return spws


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

# Intended for use with single trigger files.
# If channels can be restricted to a smaller number (e.g. just the detection channels),
# then this can probably all be done in memory on the server for a whole experiment.
def spw_explorer_np(
    time,
    hpc_lfps,
    hpc_csd,
    spws,
    axes,
    window_length=1,
    event_number=0,
):

    for ax in axes:
        ax.cla()

    # Compute peri-event window
    spw = spws.loc[event_number]
    window_start_time = spw.peak_time - window_length / 2
    window_end_time = spw.peak_time + window_length / 2

    window_mask = np.logical_and(time >= window_start_time, time <= window_end_time)
    hpc_lfps = hpc_lfps[window_mask]
    hpc_csd = hpc_csd[window_mask]
    time = time[window_mask]

    # Plot CSD
    colormesh_explorer(time, hpc_csd, ax=axes[0])

    # Plot raw LFPs
    lfp_explorer(time, hpc_lfps, axes[1])
    axes[1].set_xlim(axes[0].get_xlim())
    axes[0].set_xlabel("Time [sec]")
    axes[0].set_ylabel("Depth (mm)")

    # Highlight each spw
    for spw in spws.itertuples():
        if (spw.start_time >= window_start_time) and (spw.end_time <= window_end_time):
            axes[0].axvspan(
                spw.start_time, spw.end_time, alpha=0.1, color="lightgrey", zorder=1000
            )
            axes[1].axvspan(
                spw.start_time, spw.end_time, alpha=0.1, color="lightgrey", zorder=1000
            )


def spw_explorer_xr(
    lfp,
    csd,
    chans,
    spws,
    axes,
    plot_duration=1,
    event_number=0,
):

    for ax in axes:
        ax.cla()

    # Compute peri-event window
    spw = spws.loc[event_number]
    plot_start_time = spw.peak_time - plot_duration / 2
    plot_end_time = spw.peak_time + plot_duration / 2

    _lfp = lfp.sel(time=slice(plot_start_time, plot_end_time), channel=chans)
    _csd = csd.swap_dims({"pos": "channel"}).sel(
        time=slice(plot_start_time, plot_end_time), channel=chans
    )

    # Plot CSD
    colormesh_explorer(
        _csd.time.values, _csd.values.T, y=_csd.pos.values, ax=axes[0], flip_dv=True
    )

    # Plot raw LFPs
    lfp_explorer(
        _lfp.time.values, _lfp.values, chan_labels=_lfp.channel.values, ax=axes[1]
    )
    axes[1].set_xlim(axes[0].get_xlim())
    axes[0].set_xlabel("Time [sec]")
    axes[0].set_ylabel("Depth (mm)")

    # Highlight each spw
    for spw in spws.itertuples():
        if (spw.start_time >= plot_start_time) and (spw.end_time <= plot_end_time):
            axes[0].axvspan(
                spw.start_time, spw.end_time, alpha=0.1, color="lightgrey", zorder=1000
            )
            axes[1].axvspan(
                spw.start_time, spw.end_time, alpha=0.1, color="lightgrey", zorder=1000
            )


# Only works for single trigger files
# I don't think lazy loading is even necessary now that we have the server.
def lazy_spw_explorer(
    spws,
    lfp_path,
    csd_params,
    axes,
    window_length=1,
    event_number=0,
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
    spw = spws.loc[event_number]
    window_start_time = spw.peak_time - window_length / 2
    window_end_time = spw.peak_time + window_length / 2

    # Load the peri-event data
    lfp = load_trigger(
        lfp_path,
        csd_params["csd_channels"],
        start_time=window_start_time,
        end_time=window_end_time,
    )

    # Select further subset of data selected for plotting
    n_data_chans = lfp.channel.size
    n_plot_chans = n_data_chans if n_plot_chans is None else n_plot_chans
    if (i_chan + n_plot_chans) > n_data_chans:
        i_chan = n_data_chans - n_plot_chans

    lfps = lfp.values[:, i_chan : i_chan + n_plot_chans]
    chan_labels = lfp.channel.values[i_chan : i_chan + n_plot_chans]

    # Plot raw LFPs
    lfp_ax.set_facecolor("none")
    if show_lfps:
        lfp_explorer(
            lfp.time.values,
            lfps,
            lfp_ax,
            chan_labels=chan_labels,
            vspace=vspace,
            zero_mean=True,
        )
        lfp_ax.margins(y=0)

    # Compute CSD
    if show_csd:
        csd = get_kcsd(
            lfp,
            np.asarray(csd_params["ele_pos"]).squeeze(),
            drop_chans=csd_params["channels_omitted_from_csd_estimation"],
            do_lcurve=False,
            gdx=csd_params["gdx"],
            R_init=csd_params["R"],
            lambd=csd_params["lambd"],
        )

    # Plot CSD
    csd_ax.set_zorder(lfp_ax.get_zorder() - 1)
    csd_ax.set_xlabel("Time [sec]")
    if show_csd:
        colormesh_explorer(csd.time.values, csd.values.T, csd_ax)
        csd_ax.set_ylabel("Depth (mm)")

    # Highlight each spw
    for spw in spws.itertuples():
        if (spw.start_time >= window_start_time) and (spw.end_time <= window_end_time):
            axes[0].axvspan(
                spw.start_time, spw.end_time, alpha=0.1, color="lightgrey", zorder=1000
            )
            axes[1].axvspan(
                spw.start_time, spw.end_time, alpha=0.1, color="lightgrey", zorder=1000
            )


def interactive_lazy_spw_explorer(spws, metadata, subject, condition, figsize=(20, 8)):
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
        lazy_spw_explorer,
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
