import numpy as np
from neurodsp.plts.utils import check_ax
from neurodsp.spectral.utils import trim_spectrogram

from ecephys.scoring import filter_states

state_colors = {
    'Wake': 'palegreen',
    'N1': 'thistle',
    'N2': 'plum',
    'REM': 'bisque'
}

def plot_spectrogram(freqs, spg_times, spg,
                     f_range=None, t_range=None,
                     yscale='linear', figsize=(18, 6), ax=None):
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
    ax.pcolormesh(spg_times, freqs, np.log10(spg), shading='gouraud')
    ax.set_yscale(yscale)
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel('Time [sec]')

    if yscale == 'log':
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
        ax.axvspan(bout.start_time, bout.end_time,
                   alpha=0.3, color=state_colors[bout.state], zorder=1000)

    ax.set_xlim(xlim)



