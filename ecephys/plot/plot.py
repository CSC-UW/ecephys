import numpy as np
from neurodsp.plts.utils import check_ax
from neurodsp.spectral.utils import trim_spectrogram
import matplotlib.pyplot as plt

from ecephys.scoring import filter_states
from ecephys.signal.timefrequency import get_perievent_cwtm

state_colors = {"Wake": "palegreen", "N1": "thistle", "N2": "plum", "REM": "bisque"}


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
    time, lfps, filtered_lfps, ripple_times, window_length=1, ripple_number=1, ax=None
):
    """Plot a ripple and peri-event data.

    Parameters
    ----------
    time: array_like, shape (n_time,)
    lfps: array_like, shape (n_time, n_signals)
        Raw LFPs used for ripple detection.
    filtered_lfps : array_like, shape (n_time, n_signals)
        The bandpass filtered LFPs used for detection.
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
    window_start_time = ripple.center_time - window_length * 1000 / 2
    window_end_time = ripple.center_time + window_length * 1000 / 2
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
    cwtm = get_perievent_cwtm(lfps[samples_to_plot], 2500, freq)
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
