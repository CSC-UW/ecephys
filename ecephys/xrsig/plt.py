import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from . import core
import ecephys


def plot_yx_channel_vectors(
    da, dim, continuous_colors=False, add_legend=False, ax=None
):

    ax = ecephys.plot.check_ax(ax, figsize=(10, 15))

    if continuous_colors:
        n = len(np.unique(da[dim]))
        cmap = plt.get_cmap("cividis")
        cmap_idx = np.linspace(0, len(cmap.colors) - 1, n).astype("int")
        color_lut = {k: cmap.colors[i] for k, i in zip(np.unique(da[dim]), cmap_idx)}
    else:
        color_lut = dict(zip(np.unique(da[dim]), sns.color_palette("colorblind")))

    for c in da[dim].values:
        dat = core.LaminarScalars(da.sel({dim: c}))
        dat.plot_laminar(ax=ax, color=color_lut[c])

    if add_legend:
        legend_elements = [
            mpatches.Patch(label=key, facecolor=color)
            for key, color in color_lut.items()
        ]
        lgd = ax.legend(
            handles=legend_elements,
            handlelength=1,
            loc="upper center",
            ncol=len(legend_elements),
            shadow=True,
            bbox_to_anchor=(0.5, 1.0),
        )
    else:
        lgd = None

    return ax, lgd
