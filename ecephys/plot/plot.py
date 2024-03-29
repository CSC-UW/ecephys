from math import gcd

import colorcet
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from ecephys.npsig.utils import mean_subtract
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
from matplotlib.ticker import MaxNLocator


_colorblind = sns.color_palette("colorblind")
_deep = sns.color_palette("deep")
_dark = sns.color_palette("dark")
_muted = sns.color_palette("muted")
_pastel = sns.color_palette("muted")
state_colors = {
    "Wake": _colorblind[7],
    "W": _colorblind[7],
    "Arousal": _pastel[7],
    "MA": _pastel[7],
    "aWk": _colorblind[7],
    "qWk": _dark[7],
    "QWK": _dark[7],
    "M": "darkseagreen",
    "Trans": "gainsboro",
    "NREM": _colorblind[2],
    "N1": _muted[2],
    "N2": _colorblind[2],
    "IS": _dark[5],
    "REM": _colorblind[1],
    "Art": "crimson",
    "Artifact": "crimson",
    "artifact": "crimson",
    "?": "crimson",
    "None": _colorblind[8],
    "NoData": "black",
    "NoDataWake": "black",
    "flat": "black",
    "Other": _colorblind[8],
    "Drug": "white",
}

_pub_wake = "white"
_pub_nrem = "lightskyblue"
_pub_rem = "orangered"
_pub_nodata = "black"
_pub_unscored = "pink"
publication_colors = {
    "Wake": _pub_wake,
    "W": _pub_wake,
    "aWk": _pub_wake,
    "qWk": _pub_wake,
    "QWK": _pub_wake,
    "Arousal": _pub_nrem,
    "MA": _pub_nrem,
    "Trans": _pub_nrem,
    "NREM": _pub_nrem,
    "N1": _pub_nrem,
    "N2": _pub_nrem,
    "IS": _pub_nrem,
    "REM": _pub_rem,
    "Art": _pub_nodata,
    "Artifact": _pub_nodata,
    "artifact": _pub_nodata,
    "None": _pub_unscored,
    "Other": _pub_unscored,
    "NoData": _pub_nodata,
    "NoDataWake": _pub_wake,
    "flat": _pub_nodata,
}

on_off_colors = {
    "on": "tomato",
    "off": "plum",
}


# This function is taken directly from neurodsp.plts.utils.
# We cannot use the neurodsp package, because a critical IBL library shadows the name.
def check_ax(ax, figsize=None):
    """Check whether a figure axes object is defined, define if not.
    Parameters
    ----------
    ax : matplotlib.Axes or None
        Axes object to check if is defined.
    Returns
    -------
    ax : matplotlib.Axes
        Figure axes object to use.
    """

    if not ax:
        _, ax = plt.subplots(figsize=figsize)

    return ax


def set_yticklabels_from_values(ylabels, ys, ax):
    # https://matplotlib.org/stable/gallery/ticks/tick_labels_from_values.html
    def format_fn(tick_val, tick_pos):
        if int(tick_val) in ys:
            return ylabels[int(tick_val)]
        else:
            return ""

    # A FuncFormatter is created automatically.
    ax.yaxis.set_major_formatter(format_fn)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))


def plot_spike_train(data, Tmax=None, ax=None, linewidth=0.1, linelengths=0.95, lineoffsets=1.0, **kwargs):
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


def plot_psth_heatmap(psth_array, ylabels, window, binsize, clim=None, cbar_label=None, ax=None):
    """PSTH Heatmap.

    Args:
    psth_array (nd-array): (nclusters x nbins) evoked rates
    ylabels (list-like): Label for each row.
    window (list-like): (- plot_before, p[ot_after) in sec
    binsize (float): Size of bins in sec
    """
    if clim is None:
        vmin, vmax = None, None
    else:
        vmin, vmax = clim

    if ax is None:
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
    xtic_len = gcd(int(abs(window[0] * 1000)), int(window[1] * 1000))
    xtic_labels = range(int(window[0] * 1000), int(window[1] * 1000) + xtic_len, xtic_len)
    xtic_locs = [(j - (window[0] * 1000)) / (binsize * 1000) for j in xtic_labels]
    if 0 not in xtic_labels:
        xtic_labels.append(0)
        xtic_locs.append(-window[0] / binsize)
    ax.set_xticks(xtic_locs)
    ax.set_xticklabels(xtic_labels, rotation=0)

    # vertical line at t=0
    plt.axvline((-window[0]) / binsize, color="black", linestyle="--", linewidth=2)

    return ax


# This function is taken directly from neurodsp.spectral.utils.
# We cannot use the neurodsp package, because a critical IBL library shadows the name.
def trim_spectrogram(freqs, times, spg, f_range=None, t_range=None):
    """Extract a frequency or time range of interest from a spectrogram.
    Parameters
    ----------
    freqs : 1d array
        Frequency values for the spectrogram.
    times : 1d array
        Time values for the spectrogram.
    spg : 2d array
        Spectrogram, or time frequency representation of a signal.
        Formatted as [n_freqs, n_time_windows].
    f_range : list of [float, float]
        Frequency range to restrict to, as [f_low, f_high].
    t_range : list of [float, float]
        Time range to restrict to, as [t_low, t_high].
    Returns
    -------
    freqs_ext : 1d array
        Extracted frequency values for the power spectrum.
    times_ext : 1d array
        Extracted segment time values
    spg_ext : 2d array
        Extracted spectrogram values.
    Notes
    -----
    This function extracts frequency ranges >= f_low and <= f_high,
    and time ranges >= t_low and <= t_high. It does not round to below
    or above f_low and f_high, or t_low and t_high, respectively.
    Examples
    --------
    Trim the spectrogram of a simulated time series:
    >>> from neurodsp.sim import sim_combined
    >>> from neurodsp.timefrequency import compute_wavelet_transform
    >>> from neurodsp.utils.data import create_times, create_freqs
    >>> fs = 500
    >>> n_seconds = 10
    >>> times = create_times(n_seconds, fs)
    >>> sig = sim_combined(n_seconds, fs,
    ...                    components={'sim_powerlaw': {}, 'sim_oscillation' : {'freq': 10}})
    >>> freqs = create_freqs(1, 15)
    >>> mwt = compute_wavelet_transform(sig, fs, freqs)
    >>> spg = abs(mwt)**2
    >>> freqs_ext, times_ext, spg_ext = trim_spectrogram(freqs, times, spg,
    ...                                                  f_range=[8, 12], t_range=[0, 5])
    """

    # Initialize spg_ext, to define for case in which neither f_range nor t_range is defined
    spg_ext = spg

    # Restrict frequency range of the spectrogram
    if f_range is not None:
        f_mask = np.logical_and(freqs >= f_range[0], freqs <= f_range[1])
        freqs_ext = freqs[f_mask]
        spg_ext = spg_ext[f_mask, :]
    else:
        freqs_ext = freqs

    # Restrict time range of the spectrogram
    if t_range is not None:
        times_mask = np.logical_and(times >= t_range[0], times <= t_range[1])
        times_ext = times[times_mask]
        spg_ext = spg_ext[:, times_mask]
    else:
        times_ext = times

    return freqs_ext, times_ext, spg_ext


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
    # NOTE: No, you cannot use imshow unless frequencies are evenly spaced!
    freqs, spg_times, spg = trim_spectrogram(freqs, spg_times, spg, f_range, t_range)

    ax = check_ax(ax, figsize=figsize)
    ax.pcolormesh(spg_times, freqs, np.log10(spg), shading="gouraud", cmap="viridis")
    ax.set_yscale(yscale)
    ax.set_ylabel("Frequency [Hz]")
    ax.set_xlabel("Time [sec]")

    if yscale == "log":
        ax.set_ylim(np.min(freqs[freqs > 0]), np.max(freqs))


def plot_heatmap(
    array,
    times,
    ylabels=None,
    figsize=(18, 6),
    ax=None,
    **kwargs,
):
    """Heatmap supporting time vector and a masked color array"""
    ax = check_ax(ax, figsize=figsize)

    plot = ax.pcolormesh(times, np.arange(array.shape[0]), array, **kwargs)
    ax.set_facecolor("black")

    plt.colorbar(plot)

    if (
        ylabels is not None
    ):  # Tick labels from values: https://matplotlib.org/stable/gallery/ticks/tick_labels_from_values.html
        set_yticklabels_from_values(ylabels, range(array.shape[0]), ax)

    return ax


def plot_on_off_overlay(on_off_df, state_colors=on_off_colors, **kwargs):
    plot_hypnogram_overlay(on_off_df, state_colors=state_colors, **kwargs)


def plot_hypnogram_overlay(
    hypnogram,
    state_colors=state_colors,
    t1_column="start_time",
    t2_column="end_time",
    ax=None,
    xlim="ax",
    ymin=0,
    ymax=1,
    figsize=(18, 3),
    alpha=0.3,
    zorder=-1,
) -> plt.Axes:
    """Shade plot background using hypnogram state.

    Parameters
    ----------
    hypnogram: pandas.DataFrame
        Hypnogram with with state, start_time, end_time columns.
    ax: matplotlib.Axes, optional
        An axes upon which to plot.
    xlim: str
        "ax" or "hg"

    Examples
    --------
    >>> fig, ax = plt.subplots()
    Overlay into background, span whole plot yaxis
    >>> ax = plot_hypnogram(hg, ax=ax)
    Overlay into foreground, outside of plot yaxis
    >>> ax = plot_hypnogram(hg, ax=ax, zorder=1000, ymin=1, ymax=1.1)
    """
    ax = check_ax(ax, figsize=figsize)

    if (xlim in ["ax", None]) and (not ax is None):
        xlim = ax.get_xlim()
    elif xlim == "hg":
        xlim = (hypnogram[t1_column].min(), hypnogram[t2_column].max())
    elif not isinstance(xlim, tuple):
        raise ValueError("Invalid value for `xlim` kwarg. Expected 'ax', 'hg' or a tuple")

    for i, bout in hypnogram.iterrows():
        # clip manually on xaxis so we can set clip_on=False for yaxis
        t1 = max(bout[t1_column], xlim[0])
        t2 = min(bout[t2_column], xlim[1])

        # Don't display out of range
        if t1 > xlim[1] or t2 < xlim[0]:
            continue

        ax.axvspan(
            t1,
            t2,
            alpha=alpha,
            color=state_colors[bout.state],
            zorder=zorder,
            ec="none",
            ymin=ymin,
            ymax=ymax,
            clip_on=False,
        )

    if xlim:
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


# Is this function still relevant?
def plot_channel_coords(chans, x, y, figsize=(4, 30)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(x, y, marker=".")

    for i, txt in enumerate(chans):
        ax.annotate(txt, (x[i] + 2, y[i]), fontsize=8)

    ax.set_xlim([0, 70])


def lfp_explorer(
    time,
    lfps,
    ax,
    chan_labels=None,
    chan_colors=None,
    window_length=None,
    window_start=None,
    n_plot_chans=None,
    i_chan=0,
    vspace=300,
    zero_mean=True,
    flip_dv=False,
):
    """Plot a static image of selected LFPs.

    Parameters
    ==========
    time: (n_time,)
        LFP timestamps
    lfps: (n_time, n_chans)
    ax: matplotlib.Axes
        Axis object on which to plot.
    chan_labels: (n_chans,) np.array
        chan_labels[i] is the label of lfp[:, i]
    window_length: float > 0
        Duration to plot, in the same units as `time`
    window_start: float
        Time of the the start of the plot, in units of `time`
    n_plot_chans: int
        The number of channels to plot
    i_chan: int
        Index of the first channel to plot (e.g. plot lfps[:, i_chan : i_chan + n_plot_chans]
    vspace: float
        Spacing between LFP traces, in units of lfps (e.g. uV)
    zero_mean: bool
        Whether to zero-mean each channel before plotting
    flip_dv: bool
        Whether to flip the dorsal-ventral axis when plotting.
    """

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
    if chan_colors is not None:
        for idx, line in enumerate(ax.lines):
            line.set_color(chan_colors[idx + i_chan])

    if chan_labels is None:
        chan_labels = np.arange(0, n_data_chans, 1)

    ax.set_yticks(lfp_centers[0, :].tolist())
    ax.set_yticklabels(chan_labels[i_chan : i_chan + n_plot_chans])
    ax.set_ylabel("Channel")


def interactive_lfp_explorer(time, lfps, chan_labels=None, figsize=(20, 8)):
    """Browse LFPs using an interactive GUI

    Parameters:
    ===========
    See `lfp_explorer`.
    """
    # Create interactive widgets for controlling plot parameters
    window_length = FloatSlider(min=0.25, max=4.0, step=0.25, value=1.0, description="Secs")
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
    jslink((window_start, "value"), (_window_start, "value"))  # Allow control from either widget for easy navigation
    n_plot_chans = IntSlider(min=1, max=lfps.shape[1], step=1, value=16, description="nCh")
    i_chan = IntSlider(min=0, max=(lfps.shape[1] - 1), step=1, value=1, description="Ch")
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
        lfp_explorer,
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


# TODO: Use imshow. Much faster
def colormesh_explorer(
    time,
    sig,
    ax,
    y=None,
    window_length=None,
    window_start=None,
    n_y=None,
    i_y=0,
    zero_mean=False,
    flip_ud=False,
):
    """Plot a static colormesh (e.g. a spectrogram or a CSD).

    Parameters
    ==========
    time: (n_time,)
        LFP timestamps
    sig: (n_time, n_y)
        The data to plot
    ax: matplotlib.Axes
        Axis object on which to plot.
    y = (n_y,)
        y[i] labels sig[:, i]. Might be a depth, frequency, etc.
    window_length: float > 0
        Duration to plot, in the same units as `time`
    window_start: float
        Time of the the start of the plot, in units of `time`
    n_y: int
        The number of y-axis values to plot.
    i_y: int
        Index of the first y-axis values to plot (e.g. plot sig[:, i_y : i_y + n_y]
    zero_mean: bool
        Whether to zero-mean each y-axis element before plotting
    flip_dv: bool
        Whether to flip the y axis when plotting.
    """
    ax.cla()

    window_length = window_length or (np.max(time) - np.min(time))
    window_start = window_start if window_start is not None else np.min(time)
    window_end = window_start + window_length
    selected_samples = np.logical_and(time >= window_start, time <= window_end)

    n_data_rows = sig.shape[1]
    y = y if y is not None else np.linspace(0, 1, n_data_rows)
    n_y = n_y if n_y is not None else n_data_rows
    if (i_y + n_y) > n_data_rows:
        i_y = n_data_rows - n_y

    sig = sig[selected_samples, i_y : i_y + n_y]
    time = time[selected_samples]
    y = y[i_y : i_y + n_y]

    if zero_mean:
        sig = mean_subtract(sig)

    ax.pcolormesh(time, y, sig.T, shading="gouraud")
    ax.set_xlim([window_start, window_end])
    ax.set_xlabel("Time [sec]")
    if flip_ud:
        ax.invert_yaxis()


def interactive_colormesh_explorer(time, sig, y=None, figsize=(20, 8)):
    """Browse a colormesh (e.g. Spectrogram or CSD) using an interactive GUI

    Parameters:
    ===========
    See `colormesh_explorer`.
    """
    # Create interactive widgets for controlling plot parameters
    window_length = FloatSlider(min=0.25, max=4.0, step=0.25, value=1.0, description="Secs")
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
    jslink((window_start, "value"), (_window_start, "value"))  # Allow control from either widget for easy navigation
    n_y = IntSlider(min=1, max=sig.shape[1], step=1, value=sig.shape[1], description="nRows")
    i_y = IntSlider(min=0, max=(sig.shape[1] - 1), step=1, value=1, description="Row")
    zero_mean = Checkbox(False, description="Zero-mean")
    flip_ud = Checkbox(False, description="U/D")

    # Lay control widgets out horizontally
    ui = HBox(
        [
            window_length,
            _window_start,
            window_start,
            n_y,
            i_y,
            zero_mean,
            flip_ud,
        ]
    )

    # Plot and display
    _, ax = plt.subplots(figsize=figsize)
    out = interactive_output(
        colormesh_explorer,
        {
            "time": fixed(time),
            "sig": fixed(sig),
            "ax": fixed(ax),
            "y": fixed(y),
            "window_length": window_length,
            "window_start": window_start,
            "n_y": n_y,
            "i_y": i_y,
            "zero_mean": zero_mean,
            "flip_ud": flip_ud,
        },
    )

    display(ui, out)


def grouped_barplot(df, cat, subcat, val, err):
    """Like sns.barplot(data=df, x=cat, hue=subcat, y=val), but with controllable error bars using err=col_name
    See https://stackoverflow.com/questions/42017049/how-to-add-error-bars-on-a-grouped-barplot-from-a-column
    """
    u = df[cat].unique()
    x = np.arange(len(u))
    subx = df[subcat].unique()
    offsets = (np.arange(len(subx)) - np.arange(len(subx)).mean()) / (len(subx) + 1.0)
    width = np.diff(offsets).mean()
    for i, gr in enumerate(subx):
        dfg = df[df[subcat] == gr]
        plt.bar(
            x + offsets[i],
            dfg[val].values,
            width=width,
            label="{}".format(gr),
            yerr=dfg[err].values,
        )
    plt.xlabel(cat)
    plt.ylabel(val)
    plt.xticks(x, u)
    plt.legend()
    plt.show()


def color_by_category(arr: np.ndarray, palette="glasbey_dark") -> tuple[list, dict]:
    # First, get unique categories, preserving order of appearance
    _, idx = np.unique(arr, return_index=True)
    categories = arr[np.sort(idx)]

    # Get colorap
    if isinstance(palette, str):
        if palette in colorcet.palette:
            palette = colorcet.palette[palette]
        else:
            palette = plt.get_cmap(palette).colors
        palette = sns.color_palette(palette, n_colors=categories.size)

    # Assing colors
    color_map = dict(zip(categories, palette))
    arr_colored = [color_map[x] for x in arr]

    return arr_colored, color_map
