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
    Label,
    HBox,
    fixed,
    interactive_output,
    jslink,
)

from ..acute import SHARPTrack

##### Functions for adding rgba column to units


def add_structure_rgba_from_colormap(units, cmap):
    units["rgba"] = units["structure"].apply(lambda s: cmap[s])


def add_structure_rgba_from_sharptrack_atlas(units):
    cmap = SHARPTrack.get_atlas_colormap()
    add_structure_rgba_from_colormap(units, cmap)


def make_structure_cmap_from_palette(units, palette="glasbey_dark"):
    assert palette in cc.palette.keys(), "Requested palette not found."
    structure_names = units["structure"].unique()
    structure_colors = sns.color_palette(
        cc.palette[palette], n_colors=len(structure_names)
    )
    return dict(zip(structure_names, structure_colors))


def add_structure_rgba_from_palette(units, palette="glasbey_dark"):
    cmap = make_structure_cmap_from_palette(units, palette)
    add_structure_rgba_from_colormap(units, cmap)


def add_unit_rgba_from_palette(units, palette="glasbey_dark"):
    assert palette in cc.palette.keys(), "Requested palette not found."
    unit_colors = sns.color_palette(cc.palette[palette], n_colors=len(units))
    units["rgba"] = [to_rgba(rgb, 1.0) for rgb in unit_colors]


def add_uniform_rgba(units, color):
    assert is_color_like(color), "Requested color not found."
    unit_colors = [color] * len(units)
    units["rgba"] = [to_rgba(rgb, 1.0) for rgb in unit_colors]


##### Functions for basic raster plotting


def get_spike_trains_for_plotting(spikes, units, start_time, end_time):
    trains = spikes.spikes.as_trains(start_time, end_time)

    if "rgba" not in units.columns:
        add_uniform_rgba(units, "black")

    trains = units.join(trains, how="outer")
    silent = trains.trains.silent()
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


def raster_from_trains(trains, title=None, xlim=None, ax=None):
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


def raster(
    spikes,
    units,
    ax,
    plot_start,
    plot_length,
):
    ax.cla()
    plot_end = plot_start + plot_length

    trains = get_spike_trains_for_plotting(spikes, units, plot_start, plot_end)
    raster_from_trains(trains, xlim=([plot_start, plot_end]), ax=ax)
    plt.tight_layout()


def raster_with_events(
    spikes,
    units,
    events,
    ax,
    plot_start,
    plot_length,
):
    raster(spikes, units, ax, plot_start, plot_length)
    plot_end = plot_start + plot_length
    mask = ((events["t1"] >= plot_start) & (events["t1"] <= plot_end)) | (
        (events["t2"] >= plot_start) & (events["t2"] <= plot_end)
    )
    _events = events[mask]

    for evt in _events.itertuples():
        ax.axvspan(
            max(evt.t1, plot_start),
            min(evt.t2, plot_end),
            fc=to_rgba("lavender", 0.1),
            ec=to_rgba("lavender", 1.0),
        )


class InteractiveRaster:
    def __init__(self, spikes, units):
        self._plotting_function = raster
        self._spikes = spikes
        self._units = units
        self._plotting_args = {
            "spikes": fixed(self._spikes),
            "units": fixed(self._units),
        }
        self.setup_time_controls()

    def setup_time_controls(self, min_duration=1, max_duration=60):
        min_time = np.floor(self._spikes["t"].min())
        max_time = np.floor(self._spikes["t"].max()) - min_duration

        self._plot_start_slider = FloatSlider(
            min=min_time,
            max=max_time,
            step=1,
            value=min_time,
            description="t=",
        )
        self._plot_start_box = BoundedFloatText(
            min=min_time,
            max=max_time,
            step=1,
            value=min_time,
            description="t=",
        )
        jslink(
            (self._plot_start_slider, "value"), (self._plot_start_box, "value")
        )  # Allow control from either widget for easy navigation

        self._plot_length_box = BoundedFloatText(
            min=min_duration,
            max=max_duration,
            step=1,
            value=np.ceil((max_duration - min_duration) / 2),
            description="Secs",
        )

        self._plotting_args.update(
            {
                "plot_start": self._plot_start_box,
                "plot_length": self._plot_length_box,
            }
        )

    def setup_ui(self):
        # Lay control widgets out horizontally
        self._ui = HBox(
            [
                self._plot_length_box,
                self._plot_start_box,
                self._plot_start_slider,
            ]
        )

    def run(self, figsize=(20, 8)):
        self.setup_ui()

        fig, ax = plt.subplots(figsize=figsize)
        fig.canvas.header_visible = False
        fig.canvas.toolbar_visible = False
        self._plotting_args.update({"ax": fixed(ax)})
        out = interactive_output(
            self._plotting_function,
            self._plotting_args,
        )
        display(self._ui, out)


class InteractiveRasterWithEvents(InteractiveRaster):
    def __init__(self, spikes, units, events):
        super(InteractiveRasterWithEvents, self).__init__(spikes, units)
        self._plotting_function = raster_with_events
        self._events = events
        self._plotting_args.update({"events": fixed(self._events)})
        self.setup_event_controls()

    def setup_event_controls(self):
        assert self._events.index.dtype == "int64"
        if "t2" not in self._events.columns:
            self._events = self._events.copy()
            self._events["t2"] = self._events["t1"]

        self._event_box = BoundedIntText(
            value=self._events.index.min(),
            min=self._events.index.min(),
            max=self._events.index.max(),
            step=1,
            description="EVT",
        )
        self._event_description_label = Label(value=self._events.iloc[0]["description"])

        def _on_event_change(change):
            evt = change["new"]
            t1 = self._events.loc[evt, "t1"]
            desc = self._events.loc[evt, "description"]
            self._event_description_label.value = desc
            self._plot_start_box.value = t1 - self._plot_length_box.value / 2

        self._event_box.observe(_on_event_change, names="value")

    def setup_ui(self):
        # Lay control widgets out horizontally
        self._ui = HBox(
            [
                self._plot_length_box,
                self._plot_start_box,
                self._plot_start_slider,
                self._event_box,
                self._event_description_label,
            ]
        )
