from math import gcd

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from ecephys.signal.utils import mean_subtract
from IPython.display import display
from ipywidgets import (
    BoundedFloatText,
    Checkbox,
    FloatSlider,
    HBox,
    IntSlider,
    fixed,
    interactive_output,
    jslink,
)
from neurodsp.plts.utils import check_ax
from neurodsp.spectral.utils import trim_spectrogram

state_colors = {
    "Wake": "palegreen",
    "W": "palegreen",
    "Arousal": "palegreen",
    "aWk": "lightgreen",
    "qWk": "seagreen",
    "M": "darkseagreen",
    "Trans": "gainsboro",
    "NREM": "plum",
    "N1": "thistle",
    "N2": "plum",
    "IS": "burlywood",
    "REM": "bisque",
    "Art": "crimson",
    "?": "crimson",
    "None": "white",
    "Drug": "grey",
}


on_off_colors = {
    "on": "tomato",
    "off": "plum",
}


def plot_spike_train(
    data, Tmax=None, ax=None, linewidth=0.1, linelengths=0.95, lineoffsets=1.0, **kwargs
):
    """Spike raster.

    Args:
        data (array-like or list of array-like)
    """
    if ax is None:
        f, ax = plt.subplots()

    ax.eventplot(
        data,
        colors="black",
        linewidth=linewidth,
        linelengths=linelengths,
        lineoffsets=lineoffsets,
        **kwargs,
    )
    ax.set_xlim(left=0)
    if Tmax is not None:
        ax.set_xlim(right=Tmax)
    return ax


def plot_psth_hist(psth_array, window, binsize, ylabel=None, ylim=None):
    f, ax = plt.subplots()
    sns.despine(f)

    # xvalues = np.linspace(window[0], window[1], len(psth_array))  # Excelude end
    xpos = np.arange(len(psth_array))
    plt.bar(
        xpos,
        psth_array,
        color="black",
        width=1.0,
        facecolor="black",
        edgecolor="black",
    )

    plt.ylim(ylim)

    # y ticks: Only integers
    from matplotlib.ticker import MaxNLocator

    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # x ticks: Only 0, first and last value and muiltiple of gcd in between
    xtic_len = gcd(abs(window[0]), window[1])
    xtic_labels = range(window[0], window[1] + xtic_len, xtic_len)
    xtic_locs = [(j - window[0]) / binsize for j in xtic_labels]
    if 0 not in xtic_labels:
        xtic_labels.append(0)
        xtic_locs.append(-window[0] / binsize)
    ax.set_xticks(xtic_locs)
    ax.set_xticklabels(xtic_labels, rotation=0)

    plt.xlabel("Time (msec)")

    # vertical line at t=0
    plt.axvline((-window[0]) / binsize, color="red", linestyle="--", linewidth=2)

    return f, ax


def plot_psth_heatmap(psth_array, ylabels, window, binsize, clim=None, cbar_label=None):
    if clim is None:
        vmin, vmax = None, None
    else:
        vmin, vmax = clim

    f, ax = plt.subplots()

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap
    # xvalues = np.arange(window[0], window[1], binsize)
    hm = sns.heatmap(
        psth_array,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        center=0,
        robust=True,
        square=False,
        linewidths=0.0,
        cbar_kws={"shrink": 0.5, "label": cbar_label},
        yticklabels=ylabels,
        #         xticklabels=xvalues,
    )
    # ax.set_yticklabels(rotation=0)

    # Hide y ticks for which label is None
    yticks = ax.yaxis.get_major_ticks()
    for i, lbl in enumerate(ylabels):
        if lbl is None:
            yticks[i].set_visible(False)
    # Force horizontal ticks
    plt.yticks(rotation=0)

    # x ticks: Only 0, first and last value
    xtic_len = gcd(abs(window[0]), window[1])
    xtic_labels = range(window[0], window[1] + xtic_len, xtic_len)
    xtic_locs = [(j - window[0]) / binsize for j in xtic_labels]
    if 0 not in xtic_labels:
        xtic_labels.append(0)
        xtic_locs.append(-window[0] / binsize)
    ax.set_xticks(xtic_locs)
    ax.set_xticklabels(xtic_labels, rotation=0)

    # vertical line at t=0
    plt.axvline((-window[0]) / binsize, color="black", linestyle="--", linewidth=2)

    return f, ax


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


def plot_on_off_overlay(on_off_df, state_colors=on_off_colors, **kwargs):
    plot_hypnogram_overlay(on_off_df, state_colors=state_colors, **kwargs)


def plot_hypnogram_overlay(
    hypnogram,
    state_colors=state_colors,
    ax=None,
    xlim=None,
    figsize=(18, 3),
    alpha=0.3,
):
    """Shade plot background using hypnogram state.

    Parameters
    ----------
    hypnogram: pandas.DataFrame
        Hypnogram with with state, start_time, end_time columns.
    ax: matplotlib.Axes, optional
        An axes upon which to plot.
    """
    xlim = ax.get_xlim() if (ax and not xlim) else xlim

    ax = check_ax(ax, figsize=figsize)

    for bout in hypnogram.itertuples():
        ax.axvspan(
            bout.start_time,
            bout.end_time,
            alpha=alpha,
            color=state_colors[bout.state],
            zorder=1000,
            ec="none",
        )

    ax.set_xlim(xlim)
    return ax


def plot_consolidated_bouts(hypnogram, consolidated, figsize=(30, 2)):
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    plot_hypnogram_overlay(
        hypnogram,
        xlim=(hypnogram.start_time.min(), hypnogram.end_time.max()),
        ax=axes[0],
    )
    for period in consolidated:
        plot_hypnogram_overlay(period, xlim=axes[0].get_xlim(), ax=axes[1])

    axes[0].set(yticks=[])
    axes[1].set(yticks=[])

    return axes


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
