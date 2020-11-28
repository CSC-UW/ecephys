import numpy as np
from neurodsp.plts.utils import check_ax
from neurodsp.spectral.utils import trim_spectrogram
import matplotlib.pyplot as plt
from ipywidgets import (
    BoundedFloatText,
    Checkbox,
    FloatSlider,
    HBox,
    IntSlider,
    fixed,
    interact,
    interactive_output,
    jslink,
)


from ecephys.scoring import filter_states
from ecephys.signal.timefrequency import get_perievent_cwtm
from ecephys.signal.utils import mean_subtract

state_colors = {
    "Wake": "palegreen",
    "N1": "thistle",
    "N2": "plum",
    "REM": "bisque",
    "None": "White",
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


def plot_hypnogram_overlay(hypnogram, ax=None):
    """Shade plot background using hypnogram state.

    Parameters
    ----------
    hypnogram: pandas.DataFrame
        Hypnogram with with state, start_time, end_time columns.
    ax: matplotlib.Axes, optional
        An axes upon which to plot.
    """
    xlim = ax.get_xlim() if ax else (None, None)

    ax = check_ax(ax, figsize=(18, 3))

    for bout in hypnogram.itertuples():
        ax.axvspan(
            bout.start_time,
            bout.end_time,
            alpha=0.3,
            color=state_colors[bout.state],
            zorder=1000,
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


def plot_ripple(
    time,
    lfps,
    filtered_lfps,
    fs,
    ripple_times,
    window_length=1,
    ripple_number=1,
    ax=None,
):
    """Plot a ripple and peri-event data.

    Parameters
    ----------
    time: array_like, shape (n_time,)
    lfps: array_like, shape (n_time, n_signals)
        Raw LFPs used for ripple detection.
    filtered_lfps : array_like, shape (n_time, n_signals)
        The bandpass filtered LFPs used for detection.
    fs: float
        The sampling frequency of the data
    ripple_times: DataFrame, shape (n_ripples, )
        Ripple event times.
    window_length: float, default: 1
        Amount of data to plot around each window, in seconds.
    ripple_number: int
        The index into `ripple_times` of the ripple to plot.
    ax: matplotlib.Axes, optional
        Axes to use for plotting. Must contain 2 axis objects.

    Examples
    --------
    To use interactively in a JupyterLab notebook, with fast redrawing:

    from ipywidgets import interact, fixed
    _, ax = plt.subplots(2, 1, figsize=(18, 6))
    _ = interact(
        plot_ripple,
        time=fixed(time),
        lfps=fixed(lfps),
        filtered_lfps=fixed(filtered_lfps),
        fs=fixed(fs),
        ripple_times=fixed(ripple_times),
        window_length=(0.25, 2, 0.25),
        ripple_number=(1, len(ripple_times), 1),
        ax=fixed(ax),
    )
    """
    if ax is None:
        _, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(18, 6))
    else:
        ax1, ax2, ax3 = ax
        ax1.cla()
        ax2.cla()
        ax3.cla()

    ripple = ripple_times.loc[ripple_number]
    window_start_time = ripple.center_time - window_length / 2
    window_end_time = ripple.center_time + window_length / 2
    samples_to_plot = np.where(
        np.logical_and(time >= window_start_time, time <= window_end_time)
    )

    ax1.plot(time[samples_to_plot], filtered_lfps[samples_to_plot], linewidth=1)

    offset_lfps = lfps - np.full(lfps.shape, np.arange(lfps.shape[1]) * 300)
    ax2.plot(
        time[samples_to_plot],
        offset_lfps[samples_to_plot],
        color="black",
        linewidth=0.5,
    )

    freq = np.linspace(1, 300, 300)
    cwtm = get_perievent_cwtm(lfps[samples_to_plot], fs, freq)
    cwtm = cwtm / (1 / freq)[:, None]
    ax3.pcolormesh(time[samples_to_plot], freq, cwtm, cmap="viridis", shading="gouraud")
    ax3.set_xlim(ax1.get_xlim())
    ax3.axhline(150, color="k", alpha=0.5, linestyle="--")
    ax3.axhline(250, color="k", alpha=0.5, linestyle="--")

    for ripple in ripple_times.itertuples():
        if (ripple.start_time >= window_start_time) and (
            ripple.end_time <= window_end_time
        ):
            ax1.axvspan(
                ripple.start_time, ripple.end_time, alpha=0.3, color="red", zorder=1000
            )
            ax2.axvspan(
                ripple.start_time, ripple.end_time, alpha=0.3, color="red", zorder=1000
            )


def _plot_timeseries_interactive(
    time,
    sig,
    ax,
    chan_labels=None,
    window_length=1.0,
    window_start=0.0,
    n_plot_chans=16,
    i_chan=0,
    vspace=300,
    zero_mean=True,
    flip_dv=False,
):
    ax.cla()

    window_start = window_start if window_start else np.min(time)
    window_end = window_start + window_length
    selected_samples = np.logical_and(time >= window_start, time <= window_end)

    n_data_chans = sig.shape[1]
    if (i_chan + n_plot_chans) > n_data_chans:
        i_chan = n_data_chans - n_plot_chans

    sig = sig[selected_samples, i_chan : i_chan + n_plot_chans]
    time = time[selected_samples]

    if zero_mean:
        sig = mean_subtract(sig)

    sig_centers = -np.full(sig.shape, np.arange(n_plot_chans) * vspace)
    if flip_dv:
        sig_centers = -sig_centers

    sig_spaced = sig + sig_centers

    ax.plot(
        time,
        sig_spaced,
        color="black",
        linewidth=0.5,
    )
    ax.set_xlim([window_start, window_end])

    if chan_labels is None:
        chan_labels = np.arange(0, n_data_chans, 1)

    ax.set_yticks(sig_centers[0, :].tolist())
    ax.set_yticklabels(chan_labels[i_chan : i_chan + n_plot_chans])


def plot_timeseries_interactive(time, sig, chan_labels=None, figsize=(20, 8)):
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
        min=1, max=sig.shape[1], step=1, value=16, description="nCh"
    )
    i_chan = IntSlider(min=1, max=sig.shape[1], step=1, value=1, description="Ch")
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
        _plot_timeseries_interactive,
        {
            "time": fixed(time),
            "sig": fixed(sig),
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


def plot_channel_coords(chans, x, y, figsize=(4, 30)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(x, y, marker=".")

    for i, txt in enumerate(chans):
        ax.annotate(txt, (x[i] + 2, y[i]), fontsize=8)

    ax.set_xlim([0, 70])