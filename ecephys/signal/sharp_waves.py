from ecephys.utils.utils import all_equal, get_values_around
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
    """Load SPWs from disk to DataFrame.

    Parameters
    ==========
    path:
        Full path to the HDF5 file containg SPW data for one SGLX trigger file.
    use_datetime: bool
        If True, load all time fields as pandas Datetime objects.
        If False, load all time fields as seconds (type: float) from trigger file start.
    """
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
    """Get the single timeseries to be threshold for SPW detection.

    Parameters
    ==========
    csd: (time, channel) DataArray
        Current source density.
    coarse_detection_chans: DataArray.sel indexer
        Channels used for SPW detection.
        Default: Use all channels present in the CSD.
    n_file_detection_chans: int
        The (preferably odd) number of neighboring channels to average over
        when producing CSD estimates for each channel.

    Returns:
    ========
    (time,) DataArray
        The minima, at each time, of the locally smoothed CSD.
        The channel of each minimum is preserved.
    """
    _csd = (
        csd.sel(channel=coarse_detection_chans)
        .rolling(channel=n_fine_detection_chans, center=True)
        .mean()
        .dropna(dim="channel")
    )
    return -_csd[_csd.argmin(dim="channel")]


def get_peak_info(sig, spws):
    """Get properties of each SPW peak.

    Parameters
    ==========
    sig: (time,) DataArray
        The signal to extract peak amplitudes, times, and channels from.
        Probably the series used for SPW detection.
    spws: DataFrame
        The SPWs, with each SPW's start and end times.
    """
    spws = spws.copy()

    def _get_peak_info(spw):
        spw_sig = sig.sel(time=slice(spw.start_time, spw.end_time))
        peak = spw_sig[spw_sig.argmax()]
        return peak.item(), peak.time.item(), peak.channel.item()

    info = list(map(_get_peak_info, spws.itertuples()))
    spws[["peak_amplitude", "peak_time", "peak_channel"]] = info
    return spws


def get_coarse_detection_chans(peak_channel, n_coarse_detection_chans, csd_chans):
    """Given a channel around which to detect SPWs, get the neighboring channels.
    Checks to make sure that you have CSD estimates for all those channels.

    Parameters:
    ===========
    peak_channel: int
        The channel around which to detect SPWs.
    n_coarse_detection_chans: int
        An odd integer, indiciating the number of neighboring channels (inclusive)
        to use for detecting SPWs.
    csd_chans: np.array
        The channels for which you have CSD estimates.
    """
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
    """Detect SPWs using thresholds whose absolute values are provided.
    Return SPWs as a DataFrame.

    Parameters
    ==========
    csd: (time, channels) DataArray
        The CSD to use for detection.
    initial_peak_channel: int
        A channel around which to start detecting SPWs at t0.
    n_coarse_detection_chans: int
        The odd number of neighboring channels on which to detect SPWs
    detection_threshold: float
        If the detection series exceeds this value, a sharp wave is detected.
    boundary_threshold: float
        The start and end times of each SPW are defined by when the detection series drops below this value.
    minimum_duration: float
        The time that a SPW must exceed the detection threshold.
    """
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
    """See `detect_by_value`, but using zscores."""
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
    """Estimate vertical drift of a landmark over time.

    See: https://www.statsmodels.org/dev/generated/statsmodels.nonparametric.smoothers_lowess.lowess.html

    Parameters:
    ===========
    t: (n_t,) float
        The time of each position estimate.
    pos: (n_t,) float
        The vertical position of the landmark at each time in `t`.

    Returns:
    ========
    time, estm: ((n_time,), (n_time,))
        The estimated true position of the landmark at each timepoint.
        Usually, time == t
    """
    out = lowess(pos, t, **kwargs)
    return out[:, 0], out[:, 1]


def estimate_drift(spws, imec_map, frac=1 / 48, it=3):
    """Estimate drift using SPW peak locations over time.

    spws: DataFrame
        SPWs to use for estimation.
    imec_map: sglx.Map
        The imec map used in the recording from which SPWs were detected.
    frac: float between (0, 1)
        The fraction of data used when estimating each position.
    it: int
        The number of residual based reweightings to perform during LOWESS estimation.
        Computation time is a linear function of this valuable, but estimate quality is not.
    """
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
    """Get epocs during which drift amount is estimated to be less than the spacing between electrodes."""
    cols = ["nearest_y", "nearest_id", "shifts"]
    dfs = list(get_epocs(drift, col, "dt") for col in cols)
    assert all_arrays_equal(
        df.index for df in dfs
    ), "Different columns yielded different epochs."

    return pd.concat(dfs, axis=1)


def get_shift_timeseries(epocs, times):
    """Get the shifts that must be applied to the raw data to correct for estimated drift."""
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
def spw_explorer_xr(
    lfp,
    csd,
    spws,
    axes,
    plot_duration=1,
    event_number=0,
    center_chan=None,
    n_chans=5,
    vspace=300,
    show_lfps=True,
    show_csd=True,
):
    """Plot a static image of an SPW, with both LFP and CSD."""

    lfp_ax, csd_ax = axes
    lfp_ax.cla()
    csd_ax.cla()

    # Compute peri-event window
    spw = spws.loc[event_number]
    plot_start_time = spw.peak_time - plot_duration / 2
    plot_end_time = spw.peak_time + plot_duration / 2

    center_chan = spw.peak_channel if center_chan is None else center_chan
    chans = get_values_around(lfp.channel.values, center_chan, n_chans)

    _lfp = lfp.sel(time=slice(plot_start_time, plot_end_time), channel=chans)
    _csd = csd.swap_dims({"pos": "channel"}).sel(
        time=slice(plot_start_time, plot_end_time), channel=chans
    )

    # Plot LFP
    lfp_ax.set_facecolor("none")
    if show_lfps:
        lfp_explorer(
            _lfp.time.values,
            _lfp.values,
            chan_labels=_lfp.channel.values,
            vspace=vspace,
            zero_mean=True,
            ax=lfp_ax,
        )
        lfp_ax.margins(y=0)
        lfp_ax.set_xlabel("Time [sec]")

    # Plot CSD
    csd_ax.set_zorder(lfp_ax.get_zorder() - 1)
    if show_csd:
        colormesh_explorer(
            _csd.time.values, _csd.values.T, y=_csd.pos.values, ax=csd_ax, flip_ud=True
        )
        csd_ax.set_ylabel("Depth (mm)")
        csd_ax.set_xlabel("Time [sec]")

    # Highlight each spw
    for spw in spws.itertuples():
        if (spw.start_time >= plot_start_time) and (spw.end_time <= plot_end_time):
            lfp_ax.annotate(
                "",
                xy=(spw.start_time, 0),
                xycoords=("data", "axes fraction"),
                xytext=(spw.end_time, 0),
                arrowprops=dict(arrowstyle="|-|", color="red", alpha=0.5),
            )
            lfp_ax.annotate(
                "",
                xy=(spw.start_time, 1),
                xycoords=("data", "axes fraction"),
                xytext=(spw.end_time, 1),
                arrowprops=dict(arrowstyle="|-|", color="red", alpha=0.5),
            )


def interactive_spw_explorer(lfp, csd, spws, figsize=(20, 8)):
    """Use ipywidgets and ipympl to create a GUI for plotting SPWs."""
    assert all(
        lfp.channel.values == csd.channel.values
    ), "LFP and CSD channels must match."

    # Create interactive widgets for controlling plot parameters
    plot_duration = FloatSlider(
        min=0.25, max=4.0, step=0.25, value=1.0, description="Sec"
    )
    event_number = SelectionSlider(
        options=spws.index.values, value=spws.index.values[0], description="SPW #"
    )
    center_chan = SelectionSlider(
        options=lfp.channel.values,
        value=spws.iloc[0].peak_channel,
        description="Center Ch.",
    )
    n_chans = BoundedIntText(
        min=1, max=lfp.channel.size, step=2, value=15, description="nCh."
    )
    vspace = BoundedIntText(min=0, max=1000000, value=300, description="Spacing")
    show_lfps = Checkbox(value=True, description="LFP")
    show_csd = Checkbox(value=True, description="CSD")

    # Lay control widgets out horizontally
    ui = HBox(
        [
            plot_duration,
            event_number,
            center_chan,
            n_chans,
            vspace,
            show_lfps,
            show_csd,
        ]
    )

    # Plot and display
    _, ax = plt.subplots(figsize=figsize)
    (lfp_ax, csd_ax) = (ax, ax.twinx())
    out = interactive_output(
        spw_explorer_xr,
        {
            "lfp": fixed(lfp),
            "csd": fixed(csd),
            "spws": fixed(spws),
            "axes": fixed((lfp_ax, csd_ax)),
            "plot_duration": plot_duration,
            "event_number": event_number,
            "center_chan": center_chan,
            "n_chans": n_chans,
            "vspace": vspace,
            "show_lfps": show_lfps,
            "show_csd": show_csd,
        },
    )

    display(ui, out)


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
    """Like spw_explorer_xr, but loads data lazily from disk, rather than requiring it
    all to be in memory before calling the function. Useful for weak PCs and laptops, but
    slower."""
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


# Likely deprecated
def interactive_lazy_spw_explorer(spws, metadata, subject, condition, figsize=(20, 8)):
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


# Is this function still used?
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
