import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from ecephys import xrsig


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


def add_structure_boundaries_to_laminar_plot(
    da: xr.DataArray,
    ax: plt.Axes,
    sigdim: str = "channel",
    struct_coord: str = "structure",
):
    boundaries = da.isel({sigdim: get_boundary_ilocs(da, coord)})
    ax.set_yticks(boundaries[sigdim])
    ax.set_yticklabels(boundaries[coord].values)
    for ch in boundaries[sigdim]:
        ax.axhline(ch, alpha=0.5, color="dimgrey", linestyle="--")


def get_boundary_ilocs(da: xr.DataArray, coord_name: str) -> np.ndarray:
    df = da[coord_name].to_dataframe()  # Just so we can use pandas utils
    df = df.loc[:, ~df.columns.duplicated()].copy()  # Drop duplicate columns
    df = df.reset_index()  # So we can find boundaries of dimensions too
    changed = df[coord_name].ne(df[coord_name].shift().bfill())
    boundary_locs = df[coord_name][changed.shift(-1, fill_value=True)].index
    return np.where(np.isin(df.index, boundary_locs))[0]
