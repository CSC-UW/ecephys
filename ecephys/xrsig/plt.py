import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from ecephys import xrsig
from ecephys import plot as eplt


def plot_laminar_scalars_vertical(
    da: xr.DataArray,
    sigdim: str = "channel",
    lamdim: str = "y",
    ax: plt.Axes = None,
    figsize=(10, 15),
    show_channel_ids=True,
    tick_params=dict(axis="y", labelsize=8),
    **line_kwargs
):
    """Plot a depth profile of values.
    Requires 'y' coordinate on 'channel' dimension."""
    xrsig.validate_laminar(da, sigdim, lamdim)
    da = da.sortby(lamdim)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    tick_labels = list(np.round(da[lamdim].values, 1))
    if show_channel_ids:
        tick_labels = list(zip(tick_labels, da[sigdim].values))

    da.plot.line(y=lamdim, ax=ax, **line_kwargs)
    ax.set_yticks(da[lamdim].values)
    ax.set_yticklabels(tick_labels)
    ax.tick_params(**tick_params)


def plot_laminar_scalars_horizontal(
    da: xr.DataArray,
    sigdim: str = "channel",
    lamdim: str = "y",
    ax: plt.Axes = None,
    figsize=(32, 10),
    show_channel_ids=True,
    tick_params=dict(axis="x", labelsize=8, labelrotation=90),
    **line_kwargs
):
    """Plot a depth profile of values.
    Requires 'y' coordinate on 'channel' dimension."""
    xrsig.validate_laminar(da, sigdim, lamdim)
    da = da.sortby(lamdim)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    tick_labels = list(np.round(da[lamdim].values, 1))
    if show_channel_ids:
        tick_labels = list(zip(tick_labels, da[sigdim].values))

    da.plot.line(x=lamdim, ax=ax, **line_kwargs)
    ax.set_xticks(da[lamdim].values)
    ax.set_xticklabels(tick_labels)
    ax.tick_params(**tick_params)


def plot_laminar_image_vertical(
    da: xr.DataArray,
    sigdim: str = "channel",
    lamdim: str = "y",
    ax: plt.Axes = None,
    figsize=(4, 20),
    show_channel_ids=True,
    tick_params=dict(axis="y", labelsize=8),
    add_colorbar=False,
    **imshow_kwargs
):
    """Plot a depth profile of values.
    Requires 'y' coordinate on 'channel' dimension."""
    xrsig.validate_laminar(da, sigdim, lamdim)
    da = da.sortby(lamdim)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    tick_labels = list(np.round(da[lamdim].values, 1))
    if show_channel_ids:
        tick_labels = list(zip(tick_labels, da[sigdim].values))

    da.plot.imshow(y=lamdim, ax=ax, add_colorbar=add_colorbar, **imshow_kwargs)
    ax.set_yticks(da[lamdim].values)
    ax.set_yticklabels(tick_labels)
    ax.tick_params(**tick_params)


def plot_laminar_image_horizontal(
    da: xr.DataArray,
    sigdim: str = "channel",
    lamdim: str = "y",
    ax: plt.Axes = None,
    figsize=(32, 6),
    show_channel_ids=True,
    tick_params=dict(axis="x", labelsize=8, labelrotation=90),
    **imshow_kwargs
):
    """Plot a depth profile of values.
    Requires 'y' coordinate on 'channel' dimension."""
    xrsig.validate_laminar(da, sigdim, lamdim)
    da = da.sortby(lamdim)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    tick_labels = list(np.round(da[lamdim].values, 1))
    if show_channel_ids:
        tick_labels = list(zip(tick_labels, da[sigdim].values))

    da.plot.imshow(x=lamdim, ax=ax, **imshow_kwargs, add_colorbar=False)
    ax.set_xticks(da[lamdim].values)
    ax.set_xticklabels(tick_labels)
    ax.tick_params(**tick_params)


def plot_laminar_timeseries(
    da: xr.DataArray,
    gain=1.0,
    sigdim: str = "channel",
    lamdim: str = "y",
    ax: plt.Axes = None,
    figsize: tuple = (32, 10),
    show_channel_ids: bool = True,
    tick_params=dict(axis="y", labelsize=8),
    **line_kwargs
):
    xrsig.validate_laminar(da, sigdim, lamdim)
    xrsig.validate_2d_timeseries(da)
    da = da.sortby(lamdim)
    data = da - da.mean(dim="time")
    data = data * gain
    data = data + data.y
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    tick_labels = list(np.round(da[lamdim].values, 1))
    if show_channel_ids:
        tick_labels = list(zip(tick_labels, da[sigdim].values))

    ax.plot(data.time.values, data, color="k", linewidth=0.5)
    ax.set_yticks(da[lamdim].values)
    ax.set_yticklabels(tick_labels)
    ax.tick_params(**tick_params)


def add_structure_boundaries_to_laminar_plot(
    da: xr.DataArray,
    ax: plt.Axes,
    sigdim: str = "channel",
    struct_coord: str = "structure",
):
    boundaries = da.isel({sigdim: get_boundary_ilocs(da, struct_coord)})
    ax.set_yticks(boundaries[sigdim])
    ax.set_yticklabels(boundaries[struct_coord].values)
    for ch in boundaries[sigdim]:
        ax.axhline(ch, alpha=0.5, color="dimgrey", linestyle="--")


def get_boundary_ilocs(da: xr.DataArray, coord_name: str) -> np.ndarray:
    df = da[coord_name].to_dataframe()  # Just so we can use pandas utils
    df = df.loc[:, ~df.columns.duplicated()].copy()  # Drop duplicate columns
    df = df.reset_index()  # So we can find boundaries of dimensions too
    changed = df[coord_name].ne(df[coord_name].shift().bfill())
    boundary_locs = df[coord_name][changed.shift(-1, fill_value=True)].index
    return np.where(np.isin(df.index, boundary_locs))[0]


def plot_traces(
    da: xr.DataArray,
    chan_labels=None,
    chan_colors=None,
    palette="glasbey_dark",
    **kwargs
):
    xrsig.validate_2d_timeseries(da)
    if isinstance(chan_labels, str) and chan_labels in da["channel"].coords:
        chan_labels = da[chan_labels].values
    if isinstance(chan_colors, str) and chan_colors in da["channel"].coords:
        chan_colors, _ = eplt.color_by_category(da[chan_colors].values, palette=palette)

    eplt.lfp_explorer(
        da.time.values,
        da.values,
        chan_labels=chan_labels,
        chan_colors=chan_colors,
        **kwargs
    )
