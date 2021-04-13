import numpy as np
from kcsd import KCSD1D
from neurodsp.plts.utils import check_ax
from neurodsp.spectral.utils import trim_spectrogram
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from ipywidgets import (
    BoundedFloatText,
    BoundedIntText,
    Checkbox,
    FloatSlider,
    HBox,
    IntSlider,
    SelectionSlider,
    Select,
    fixed,
    interact,
    interactive_output,
    jslink,
)

# from ecephys.data import channel_groups, paths
from ecephys.scoring import filter_states
from ecephys.sglx_utils import load_timeseries
from ecephys.signal.sharp_wave_ripples import apply_ripple_filter
from ecephys.signal.timefrequency import get_perievent_cwtm
from ecephys.signal.utils import mean_subtract
from ecephys.signal.csd import get_kcsd

state_colors = {
    "Wake": "palegreen",
    "W": "palegreen",
    "aWk": "lightgreen",
    "qWk": "seagreen",
    "M": "darkseagreen",
    "NREM": "royalblue",
    "N1": "thistle",
    "N2": "plum",
    "REM": "magenta",
    "Art": "crimson",
    "?": "crimson",
    "None": "White",
    "Unsure": "darkorange",
    "Brief-Arousal": "palegreen",
    "Slow-During-Wake": "chartreuse",
    "Transition": "grey"
}


def plot_spectrogram(
    freqs,
    spg_times,
    spg,
    f_range=None,
    t_range=None,
    yscale="linear",
    figsize=(18, 6),
    ax=None,
):
    """Plot a spectrogram.

    Parameters
    ----------
    freqs: 1d array
        Frequencies at which spectral power was computed.
    spg_times: 1d array
        Times at which spectral power estimates are centered.
    spg: (n_freqs, n_spg_times)
        Spectrogram data
    f_range: list of [float, float]
        Frequency range to restrict to, as [f_low, f_high].
    t_range: list of [float, float]
        Time range to restrict to, as [t_low, t_high].
    yscale: str, optional, default: 'linear'
        Scaling to apply to the y-axis (e.g. 'log'). Anything pyplot will accept.
    figsize: tuple, optional, default: (18, 6)
        Size of the figure to plot
    ax: matplotlib.Axes, optional
        Axes upon which to plot.
    """
    freqs, spg_times, spg = trim_spectrogram(freqs, spg_times, spg, f_range, t_range)

    ax = check_ax(ax, figsize=figsize)
    ax.pcolormesh(spg_times, freqs, np.log10(spg), shading="gouraud")
    ax.set_yscale(yscale)
    ax.set_ylabel("Frequency [Hz]")
    ax.set_xlabel("Time [sec]")

    if yscale == "log":
        ax.set_ylim(np.min(freqs[freqs > 0]), np.max(freqs))


def plot_hypnogram_overlay(
    hypnogram, state_colors=state_colors, ax=None, figsize=(18, 3)
):
    """Shade plot background using hypnogram state.

    Parameters
    ----------
    hypnogram: pandas.DataFrame
        Hypnogram with with state, start_time, end_time columns.
    ax: matplotlib.Axes, optional
        An axes upon which to plot.
    """
    xlim = ax.get_xlim() if ax else (None, None)

    ax = check_ax(ax, figsize=figsize)

    for bout in hypnogram.itertuples():
        ax.axvspan(
            bout.start_time,
            bout.end_time,
            alpha=0.3,
            color=state_colors[bout.state],
            zorder=1000,
            ec="none",
        )

    ax.set_xlim(xlim)


def plot_all_ripples(time, lfps, filtered_lfps, ripple_times):
    """Plot an overview of all ripples detected."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 6))

    ax1.plot(time, filtered_lfps)
    for ripple in ripple_times.itertuples():
        ax1.axvspan(
            ripple.start_time, ripple.end_time, alpha=0.3, color="red", zorder=1000
        )

    ax2.plot(time, lfps)
    for ripple in ripple_times.itertuples():
        ax2.axvspan(
            ripple.start_time, ripple.end_time, alpha=0.3, color="red", zorder=1000
        )

    plt.show()


def plot_channel_coords(chans, x, y, figsize=(4, 30)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(x, y, marker=".")

    for i, txt in enumerate(chans):
        ax.annotate(txt, (x[i] + 2, y[i]), fontsize=8)

    ax.set_xlim([0, 70])


def _lfp_explorer(
    time,
    lfps,
    ax,
    chan_labels=None,
    window_length=None,
    window_start=None,
    n_plot_chans=None,
    i_chan=0,
    vspace=300,
    zero_mean=True,
    flip_dv=False,
):
    ax.cla()

    window_start = window_start or np.min(time)
    window_length = window_length or (np.max(time) - np.min(time))
    window_end = window_start + window_length
    selected_samples = np.logical_and(time >= window_start, time <= window_end)

    n_data_chans = lfps.shape[1]
    n_plot_chans = n_plot_chans or n_data_chans
    if (i_chan + n_plot_chans) > n_data_chans:
        i_chan = n_data_chans - n_plot_chans

    lfps = lfps[selected_samples, i_chan : i_chan + n_plot_chans]
    time = time[selected_samples]

    if zero_mean:
        lfps = mean_subtract(lfps)

    lfp_centers = -np.full(lfps.shape, np.arange(n_plot_chans) * vspace)
    if flip_dv:
        lfp_centers = -lfp_centers

    sig_spaced = lfps + lfp_centers

    ax.plot(
        time,
        sig_spaced,
        color="black",
        linewidth=0.5,
    )
    ax.set_xlim([window_start, window_end])

    if chan_labels is None:
        chan_labels = np.arange(0, n_data_chans, 1)

    ax.set_yticks(lfp_centers[0, :].tolist())
    ax.set_yticklabels(chan_labels[i_chan : i_chan + n_plot_chans])
    ax.set_ylabel("Channel")


def lfp_explorer(time, lfps, chan_labels=None, figsize=(20, 8)):
    # Create interactive widgets for controlling plot parameters
    window_length = FloatSlider(
        min=0.25, max=4.0, step=0.25, value=1.0, description="Secs"
    )
    window_start = FloatSlider(
        min=np.min(time),
        max=np.max(time),
        step=0.1,
        value=np.min(time),
        description="Pos",
    )
    _window_start = BoundedFloatText(
        min=np.min(time),
        max=np.max(time),
        step=0.1,
        value=np.min(time),
        description="Pos",
    )
    jslink(
        (window_start, "value"), (_window_start, "value")
    )  # Allow control from either widget for easy navigation
    n_plot_chans = IntSlider(
        min=1, max=lfps.shape[1], step=1, value=16, description="nCh"
    )
    i_chan = IntSlider(
        min=0, max=(lfps.shape[1] - 1), step=1, value=1, description="Ch"
    )
    vspace = IntSlider(min=0, max=100000, step=100, value=300, description="V Space")
    zero_mean = Checkbox(True, description="Zero-mean")
    flip_dv = Checkbox(False, description="D/V")

    # Lay control widgets out horizontally
    ui = HBox(
        [
            window_length,
            _window_start,
            window_start,
            n_plot_chans,
            i_chan,
            vspace,
            zero_mean,
            flip_dv,
        ]
    )

    # Plot and display
    _, ax = plt.subplots(figsize=figsize)
    out = interactive_output(
        _lfp_explorer,
        {
            "time": fixed(time),
            "lfps": fixed(lfps),
            "ax": fixed(ax),
            "chan_labels": fixed(chan_labels),
            "window_length": window_length,
            "window_start": window_start,
            "n_plot_chans": n_plot_chans,
            "i_chan": i_chan,
            "vspace": vspace,
            "zero_mean": zero_mean,
            "flip_dv": flip_dv,
        },
    )

    display(ui, out)


def _colormesh_timeseries_explorer(
    time,
    sig,
    ax,
    y=None,
    window_length=None,
    window_start=None,
    n_plot_rows=None,
    i_row=0,
    zero_mean=False,
    flip_dv=False,
):
    ax.cla()

    window_length = window_length or (np.max(time) - np.min(time))
    window_start = window_start if window_start is not None else np.min(time)
    window_end = window_start + window_length
    selected_samples = np.logical_and(time >= window_start, time <= window_end)

    n_data_rows = sig.shape[1]
    y = y if y is not None else np.linspace(0, 1, n_data_rows)
    n_plot_rows = n_plot_rows if n_plot_rows is not None else n_data_rows
    if (i_row + n_plot_rows) > n_data_rows:
        i_row = n_data_rows - n_plot_rows

    sig = sig[selected_samples, i_row : i_row + n_plot_rows]
    time = time[selected_samples]
    y = y[i_row : i_row + n_plot_rows]

    if zero_mean:
        sig = mean_subtract(sig)

    ax.pcolormesh(time, y, sig.T, shading="gouraud")
    ax.set_xlim([window_start, window_end])
    ax.set_xlabel("Time [sec]")
    if flip_dv:
        ax.invert_yaxis()


def colormesh_timeseries_explorer(time, sig, y=None, figsize=(20, 8)):
    # Create interactive widgets for controlling plot parameters
    window_length = FloatSlider(
        min=0.25, max=4.0, step=0.25, value=1.0, description="Secs"
    )
    window_start = FloatSlider(
        min=np.min(time),
        max=np.max(time),
        step=0.1,
        value=np.min(time),
        description="Pos",
    )
    _window_start = BoundedFloatText(
        min=np.min(time),
        max=np.max(time),
        step=0.1,
        value=np.min(time),
        description="Pos",
    )
    jslink(
        (window_start, "value"), (_window_start, "value")
    )  # Allow control from either widget for easy navigation
    n_plot_rows = IntSlider(
        min=1, max=sig.shape[1], step=1, value=sig.shape[1], description="nRows"
    )
    i_row = IntSlider(min=0, max=(sig.shape[1] - 1), step=1, value=1, description="Row")
    zero_mean = Checkbox(False, description="Zero-mean")
    flip_dv = Checkbox(False, description="D/V")

    # Lay control widgets out horizontally
    ui = HBox(
        [
            window_length,
            _window_start,
            window_start,
            n_plot_rows,
            i_row,
            zero_mean,
            flip_dv,
        ]
    )

    # Plot and display
    _, ax = plt.subplots(figsize=figsize)
    out = interactive_output(
        _colormesh_timeseries_explorer,
        {
            "time": fixed(time),
            "sig": fixed(sig),
            "ax": fixed(ax),
            "y": fixed(y),
            "window_length": window_length,
            "window_start": window_start,
            "n_plot_rows": n_plot_rows,
            "i_row": i_row,
            "zero_mean": zero_mean,
            "flip_dv": flip_dv,
        },
    )

    display(ui, out)


def _lazy_ripple_explorer(
    ripples,
    metadata,
    subject,
    condition,
    axes,
    window_length=1,
    spw_number=1,
):
    """Plot a ripple and peri-event data.

    Parameters
    ----------
    ripples: DataFrame, shape (n_ripples, )
        Each ripple must have a 'midpoint' field
    metadata: dict
        The metadata from ripple detection. Must include "chans" field containing channels used for detection.
    subject: str
        The name of the subject as stored in ecephys.data
    condition:
        The experimental condition, as stored in ecephys.data
    axes: matplotlib.Axes, optional
        Axes to use for plotting. Must contain 4 axis objects.
    window_length: float, default: 1
        Amount of data to plot around each window, in seconds.
    ripple_number: int
        The index into `ripple_times` of the ripple to plot.
    """

    for ax in axes:
        ax.cla()

    # Compute peri-event window
    ripple = ripples.loc[ripple_number]
    window_start_time = ripple.midpoint - window_length / 2
    window_end_time = ripple.midpoint + window_length / 2

    # Load the peri-event data
    all_chans = channel_groups.full[subject]
    (time, sig, fs) = load_timeseries(
        Path(paths.lfp_bin[condition][subject]),
        all_chans,
        start_time=window_start_time,
        end_time=window_end_time,
    )

    # Select subset of data used for detection and apply the same ripple filter
    idx_detection_chans = np.isin(all_chans, metadata["chans"])
    detection_sig = sig[:, idx_detection_chans]
    filtered_detection_sig = apply_ripple_filter(detection_sig, fs)

    # Plot filtered detection signal
    axes[0].plot(time, filtered_detection_sig, linewidth=1)

    # Plot raw LFPs
    offset_lfps = detection_sig - np.full(
        detection_sig.shape, np.arange(detection_sig.shape[1]) * 300
    )
    axes[1].plot(
        time,
        offset_lfps,
        color="black",
        linewidth=0.5,
    )
    axes[1].set_xlim(axes[0].get_xlim())

    # Compute CWTM
    freq = np.linspace(1, 300, 300)
    cwtm = get_perievent_cwtm(detection_sig, fs, freq, normalize=True)

    # Plot CWTM
    axes[2].pcolormesh(time, freq, cwtm, cmap="viridis", shading="gouraud")
    axes[2].set_xlim(axes[0].get_xlim())
    axes[2].axhline(150, color="k", alpha=0.5, linestyle="--")
    axes[2].axhline(250, color="k", alpha=0.5, linestyle="--")

    # Compute CSD
    n_chans = len(all_chans)
    intersite_distance = 0.020
    ele_pos = np.linspace(0.0, (n_chans - 1) * intersite_distance, n_chans).reshape(
        n_chans, 1
    )
    k = KCSD1D(ele_pos, sig.T)
    est_csd = k.values("CSD")

    # Find and select hippocampal sources
    idx_hpc_chans = np.isin(all_chans, channel_groups.hippocampus[subject])
    hpc_ele_pos = ele_pos[idx_hpc_chans]
    idx_hpc_src = np.logical_and(
        k.estm_x >= np.min(hpc_ele_pos), k.estm_x <= np.max(hpc_ele_pos)
    )

    # Plot CSD
    axes[3].pcolormesh(
        time, k.estm_x[idx_hpc_src], est_csd[idx_hpc_src, :], shading="gouraud"
    )
    axes[3].set_xlim(axes[0].get_xlim())
    axes[3].set_xlabel("Time [sec]")
    axes[3].set_ylabel("Depth (mm)")

    # Plot lines on CSD to show area used for ripple detection
    detection_ele_pos = ele_pos[idx_detection_chans]
    axes[3].axhline(np.min(detection_ele_pos), alpha=0.5, color="k", linestyle=":")
    axes[3].axhline(np.max(detection_ele_pos), alpha=0.5, color="k", linestyle=":")

    # Highlight each ripple
    for ripple in ripples.itertuples():
        if (ripple.start_time >= window_start_time) and (
            ripple.end_time <= window_end_time
        ):
            axes[0].axvspan(
                ripple.start_time, ripple.end_time, alpha=0.3, color="red", zorder=1000
            )
            axes[1].axvspan(
                ripple.start_time, ripple.end_time, alpha=0.3, color="red", zorder=1000
            )


def lazy_ripple_explorer(ripples, metadata, subject, condition, figsize=(20, 10)):
    """
    Examples
    --------
    %matplotlib widget
    lazy_ripple_explorer(ripples metadata, subject, condition)
    """
    # Create interactive widgets for controlling plot parameters
    window_length = FloatSlider(
        min=0.25, max=4.0, step=0.25, value=1.0, description="Secs"
    )
    ripple_number = IntSlider(
        min=1, max=len(ripples), step=1, value=1, description="Ripple number"
    )
    _ripple_number = BoundedIntText(
        min=1, max=len(ripples), step=1, value=1, description="Ripple number"
    )
    jslink(
        (ripple_number, "value"), (_ripple_number, "value")
    )  # Allow control from either widget for easy navigation

    # Lay control widgets out horizontally
    ui = HBox([window_length, ripple_number, _ripple_number])

    # Plot and display
    _, axes = plt.subplots(4, 1, figsize=figsize)
    out = interactive_output(
        _lazy_ripple_explorer,
        {
            "ripples": fixed(ripples),
            "metadata": fixed(metadata),
            "subject": fixed(subject),
            "condition": fixed(condition),
            "axes": fixed(axes),
            "window_length": window_length,
            "ripple_number": ripple_number,
        },
    )

    display(ui, out)


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
    (time, lfps, fs) = load_timeseries(
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