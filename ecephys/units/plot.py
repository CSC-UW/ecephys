import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import colorcet as cc
import seaborn as sns
from matplotlib.colors import to_rgba, is_color_like
from IPython.display import display
from ipywidgets import (
    BoundedFloatText,
    BoundedIntText,
    FloatSlider,
    HBox,
    fixed,
    interactive_output,
    jslink,
)

from ..acute import SHARPTrack

##### Functions for adding rgba column to units


def add_structure_rgba_from_colormap(units, cmap):
    units.df["rgba"] = units.df["structure"].apply(lambda s: cmap[s])


def add_structure_rgba_from_sharptrack_atlas(units):
    cmap = SHARPTrack.get_atlas_colormap()
    add_structure_rgba_from_colormap(units, cmap)


def make_structure_cmap_from_palette(units, palette="glasbey_dark"):
    assert palette in cc.palette.keys(), "Requested palette not found."
    structure_names = units.df["structure"].unique()
    structure_colors = sns.color_palette(
        cc.palette[palette], n_colors=len(structure_names)
    )
    return dict(zip(structure_names, structure_colors))


def add_structure_rgba_from_palette(units, palette="glasbey_dark"):
    cmap = make_structure_cmap_from_palette(units, palette)
    add_structure_rgba_from_colormap(units, cmap)


def add_unit_rgba_from_palette(units, palette="glasbey_dark"):
    assert palette in cc.palette.keys(), "Requested palette not found."
    unit_colors = sns.color_palette(cc.palette[palette], n_colors=len(units.df))
    units.df["rgba"] = [to_rgba(rgb, 1.0) for rgb in unit_colors]


def add_uniform_rgba(units, color):
    assert is_color_like(color), "Requested color not found."
    unit_colors = [color] * len(units.df)
    units.df["rgba"] = [to_rgba(rgb, 1.0) for rgb in unit_colors]


##### Functions for basic raster plotting


def get_spike_trains_for_plotting(spikes, units, start_time, end_time):
    _spikes = spikes.select_time(start_time, end_time)
    trains = _spikes.df.groupby("cluster_id")["t"].unique()

    if "rgba" not in units.df.columns:
        add_uniform_rgba(units, "black")

    trains = units.df.join(trains, how="outer")
    silent = trains["t"].isna()
    # Add ghost spikes at very start and end of window to silent trains, to reserve space for them on the plot's x and y axes.
    trains.loc[silent, "t"] = pd.Series(
        [np.array((start_time, end_time))] * sum(silent)
    ).values
    # Make silent units white and transparent, so that they are invisible.
    trains.loc[silent, "rgba"] = pd.Series([to_rgba("white", 0.0)] * sum(silent)).values
    return trains.sort_values("depth")


def _col_diff(df, col):
    return df[col].ne(df[col].shift().bfill())


def _get_boundary_unit_ilocs(trains):
    """Find trains.ilocs where trains['structure'] changes.
    These are the units that lie closest to structure boundaries, since `trains` is sorted by depth."""
    changed = _col_diff(trains, "structure")
    boundary_unit_ids = trains.structure[changed.shift(-1, fill_value=True)].index
    return np.where(np.isin(trains.index, boundary_unit_ids))[0]


def plot_spike_trains(trains, title=None, xlim=None, ax=None):
    MIN_UNITS_FOR_YTICKLABEL = 5

    if ax is None:
        fig, ax = plt.subplots(figsize=(36, 14))

    ax.eventplot(data=trains, positions="t", colors="rgba")

    if "structure" in trains.columns:
        boundary_unit_ilocs = _get_boundary_unit_ilocs(trains)
        ax.set_yticks(boundary_unit_ilocs)
        do_label = np.diff(boundary_unit_ilocs, prepend=0) > MIN_UNITS_FOR_YTICKLABEL
        ax.set_yticklabels(
            [
                trains["structure"].iloc[iloc] if label else ""
                for label, iloc in zip(do_label, boundary_unit_ilocs)
            ]
        )
    else:
        ax.set_yticks([])

    if title is not None:
        ax.set_title(title, loc="left")

    if xlim is not None:
        ax.set_xlim(xlim)

    ax.margins(x=0)


##### Functions for interactive raster plotting


def raster_explorer(
    spikes,
    units,
    ax,
    plot_start,
    plot_length,
):
    ax.cla()
    plot_end = plot_start + plot_length

    trains = get_spike_trains_for_plotting(spikes, units, plot_start, plot_end)
    plot_spike_trains(trains, xlim=([plot_start, plot_end]), ax=ax)
    plt.tight_layout()


def interactive_raster_explorer(spikes, units, figsize=(20, 8)):
    """Requires %matplotlib widget in notebook cell"""
    # Create interactive widgets for controlling plot parameters
    spikes_start = np.floor(spikes.df["t"].min())
    spikes_end = np.floor(spikes.df["t"].max())
    plot_start_slider = FloatSlider(
        min=spikes_start,
        max=spikes_end,
        step=1,
        value=spikes_start,
        description="t=",
    )
    plot_start_box = BoundedFloatText(
        min=spikes_start,
        max=spikes_end,
        step=1,
        value=spikes_start,
        description="t=",
    )
    jslink(
        (plot_start_slider, "value"), (plot_start_box, "value")
    )  # Allow control from either widget for easy navigation
    plot_length_box = BoundedFloatText(
        min=1, max=60, step=1, value=1, description="Secs"
    )

    # Lay control widgets out horizontally
    ui = HBox(
        [
            plot_length_box,
            plot_start_box,
            plot_start_slider,
        ]
    )

    # Plot and display
    fig, ax = plt.subplots(figsize=figsize)
    fig.canvas.header_visible = False
    fig.canvas.toolbar_visible = False
    out = interactive_output(
        raster_explorer,
        {
            "spikes": fixed(spikes),
            "units": fixed(units),
            "ax": fixed(ax),
            "plot_start": plot_start_box,
            "plot_length": plot_length_box,
        },
    )

    display(ui, out)


def raster_explorer_with_events(
    spikes,
    units,
    events,
    ax,
    plot_start,
    plot_length,
):
    ax.cla()
    plot_end = plot_start + plot_length

    mask = ((events["t1"] >= plot_start) & (events["t1"] <= plot_end)) | (
        (events["t2"] >= plot_start) & (events["t2"] <= plot_end)
    )
    _events = events[mask]

    trains = get_spike_trains_for_plotting(spikes, units, plot_start, plot_end)
    plot_spike_trains(trains, xlim=([plot_start, plot_end]), ax=ax)

    for evt in _events.itertuples():
        ax.axvspan(
            max(evt.t1, plot_start),
            min(evt.t2, plot_end),
            fc=to_rgba("lavender", 0.1),
            ec=to_rgba("lavender", 1.0),
        )

    plt.tight_layout()


def interactive_raster_explorer_with_events(spikes, units, events, figsize=(20, 8)):
    """Requires %matplotlib widget in notebook cell"""
    # Create interactive widgets for controlling plot parameters
    MIN_DURATION, MAX_DURATION = (1, 60)
    min_time = np.floor(spikes.df["t"].min())
    max_time = np.floor(spikes.df["t"].max()) - MIN_DURATION
    plot_start_slider = FloatSlider(
        min=min_time,
        max=max_time,
        step=1,
        value=min_time,
        description="t=",
    )
    plot_start_box = BoundedFloatText(
        min=min_time,
        max=max_time,
        step=1,
        value=min_time,
        description="t=",
    )
    jslink(
        (plot_start_slider, "value"), (plot_start_box, "value")
    )  # Allow control from either widget for easy navigation
    plot_length_box = BoundedFloatText(
        min=MIN_DURATION,
        max=MAX_DURATION,
        step=1,
        value=np.ceil((MAX_DURATION - MIN_DURATION) / 2),
        description="Secs",
    )

    # Handle events
    assert events.index.dtype == "int64"
    if "t2" not in events.columns:
        events = events.copy()
        events["t2"] = events["t1"]

    event_box = BoundedIntText(
        value=events.index.min(),
        min=events.index.min(),
        max=events.index.max(),
        step=1,
        description="EVT",
    )

    def _on_event_change(change):
        evt = change["new"]
        t1 = events.loc[evt, "t1"]
        plot_start_box.value = t1 - plot_length_box.value / 2

    event_box.observe(_on_event_change, names="value")

    # Lay control widgets out horizontally
    ui = HBox(
        [
            plot_length_box,
            plot_start_box,
            plot_start_slider,
            event_box,
        ]
    )

    # Plot and display
    fig, ax = plt.subplots(figsize=figsize)
    fig.canvas.header_visible = False
    fig.canvas.toolbar_visible = False
    out = interactive_output(
        raster_explorer_with_events,
        {
            "spikes": fixed(spikes),
            "units": fixed(units),
            "events": fixed(events),
            "ax": fixed(ax),
            "plot_start": plot_start_box,
            "plot_length": plot_length_box,
        },
    )

    display(ui, out)
