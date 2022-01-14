import xarray as xr
import numpy as np
from kcsd import KCSD1D
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks


def get_kcsd(sig, ele_pos, drop_chans=[], do_lcurve=False, **kcsd_kwargs):
    """If signal units are in uV, then CSD units are in nA/mm.

    Paramters:
    ----------
    sig: xr.DataArray (time, channel)
        The data on which to compute a 1D KCSD.
    ele_pos: (n_channels,)
        The positions, in mm, of each electrode in `sig`.
    drop_chans: list
        Channels (as they appear in `sig`) to exclude when estimating the CSD.
    do_lcurve: Boolean
        Whether to perform L-Curve parameter estimation.
    **kcsd_kwargs:
        Keywords passed to KCSD1D.

    Returns:
    --------
    csd: xr.DataArray (time, pos)
        The CSD estimates. If the estimation locations requested of KCSD1D correspond
        exactly to electrode positions, a `channel` coordinate on the `pos` dimension
        will give corresponding channels for each estimate.
    """
    channels = sig.channel.values  # Save for later
    sig = sig.assign_coords({"pos": ("channel", ele_pos)})
    sig = sig.drop_sel(channel=drop_chans, errors="ignore")
    k = KCSD1D(
        sig.pos.values.reshape(-1, 1),
        sig.transpose("channel", "time").values,
        **kcsd_kwargs
    )

    if do_lcurve:
        print("Performing L-curve parameter estimation...")
        k.L_curve()

    csd = xr.DataArray(
        k.values("CSD"),
        dims=("pos", "time"),
        coords={"pos": k.estm_x, "time": sig.time.values},
    )
    if "timedelta" in sig.coords:
        csd = csd.assign_coords({"timedelta": ("time", sig.timedelta.values)})
    if "datetime" in sig.coords:
        csd = csd.assign_coords({"datetime": ("time", sig.datetime.values)})

    if np.allclose(k.estm_x, ele_pos):
        csd = csd.assign_coords({"channel": ("pos", channels)})

    return csd.assign_attrs(kcsd=k, fs=sig.fs)


def get_epoched_minima(csd, epoch_length):
    assert csd.time.values.max() >= epoch_length, "Epochs longer than data."
    return csd.coarsen(
        time=int(csd.fs * epoch_length), boundary="trim", coord_func={"time": "min"}
    ).min()


def get_epoched_variance(csd, epoch_length):
    assert csd.time.values.max() >= epoch_length, "Epochs longer than data."
    return csd.coarsen(
        time=int(csd.fs * epoch_length), boundary="trim", coord_func={"time": "min"}
    ).var()


def plot_epoched_depth_profiles(da, figsize=(36, 8)):
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        da,
        xticklabels=da.time.values.round(),
        yticklabels=da.channel.values,
        cbar=False,
    )
    ax.set(
        xticks=ax.get_xticks()[::2],
        yticks=ax.get_yticks()[::4],
        xlabel="Epoch center time (s)",
        ylabel="Channel",
    )
    return fig, ax


def plot_depth_profile(
    da,
    figsize=(36, 5),
    ylabel=None,
    mark_positive_peaks=False,
    mark_negative_peaks=False,
):
    fig, ax = plt.subplots(figsize=figsize)
    channel_indices = np.arange(len(da.channel))
    sns.barplot(x=channel_indices, y=da, color="steelblue", ax=ax)
    ax.set(xlabel="Channel", ylabel=ylabel)
    ax.set_xticks(channel_indices)
    ax.set_xticklabels(da.channel.values, rotation=90)

    if mark_positive_peaks:
        peaks, _ = find_peaks(da.values, prominence=1000, distance=10)
        sns.scatterplot(x=peaks, y=da.values[peaks], marker="x", color="red", ax=ax)

    if mark_negative_peaks:
        peaks, _ = find_peaks(-da.values, prominence=1000, distance=10)
        sns.scatterplot(x=peaks, y=da.values[peaks], marker="x", color="red", ax=ax)

    return fig, ax
