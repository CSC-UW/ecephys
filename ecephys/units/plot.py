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
from ..utils import Bunch

##### Functions for adding rgba column to units


def map_categories_to_palette(df, col_name, palette="glasbey_dark"):
    assert palette in cc.palette.keys(), "Requested palette not found."
    category_names = df[col_name].unique()
    category_colors = sns.color_palette(
        cc.palette[palette], n_colors=len(category_names)
    )
    return dict(zip(category_names, category_colors))


def set_rgba_from_atlas(df):
    cmap = SHARPTrack.get_atlas_colormap()
    df["rgba"] = df["structure"].apply(lambda s: cmap[s])


def set_rgba_from_structure(df, palette="glasbey_dark"):
    cmap = map_categories_to_palette(df, "structure", palette)
    df["rgba"] = df["structure"].apply(lambda s: cmap[s])


def set_rgba_from_cluster(df, palette="glasbey_dark"):
    assert palette in cc.palette.keys(), "Requested palette not found."
    cluster_colors = sns.color_palette(cc.palette[palette], n_colors=len(df))
    df["rgba"] = [to_rgba(rgb, 1.0) for rgb in cluster_colors]


def set_uniform_rgba(df, color):
    assert is_color_like(color), "Requested color not found."
    cluster_colors = [color] * len(df)
    df["rgba"] = [to_rgba(rgb, 1.0) for rgb in cluster_colors]


def set_rgba_from_probe(df, palette="glasbey_dark"):
    cmap = map_categories_to_palette(df, "probe", palette)
    df["rgba"] = df["probe"].apply(lambda s: cmap[s])


##### Functions for basic raster plotting


def get_spike_trains_for_plotting(spikes, units, start_time, end_time):
    trains = spikes.spikes.as_trains(start_time, end_time)

    if "rgba" not in units.columns:
        set_uniform_rgba(units, "black")

    trains = units.join(trains, how="outer")
    silent = trains.trains.silent()
    # Add ghost spikes at very start and end of window to silent trains, to reserve space for them on the plot's x and y axes.
    trains.loc[silent, "t"] = pd.Series(
        [np.array((start_time, end_time))] * sum(silent)
    ).values
    # Make silent units white and transparent, so that they are invisible.
    trains.loc[silent, "rgba"] = pd.Series([to_rgba("white", 0.0)] * sum(silent)).values
    return trains.sort_values("depth")


def get_multiprobe_spike_trains_for_plotting(multiprobe_sorting, start_time, end_time):
    trains = Bunch()
    for probe in multiprobe_sorting:
        trains[probe] = get_spike_trains_for_plotting(
            multiprobe_sorting[probe].spikes,
            multiprobe_sorting[probe].units,
            start_time,
            end_time,
        )

    return pd.concat(
        [trains[probe] for probe in trains], keys=trains.keys(), names=["probe"]
    ).reset_index()


def _col_diff(df, col_name):
    return df[col_name].ne(df[col_name].shift().bfill())


def _get_boundary_ilocs(df, col_name):
    """Find trains.ilocs where trains['structure'] changes.
    These are the units that lie closest to structure boundaries, since `trains` is sorted by depth."""
    changed = _col_diff(df, col_name)
    boundary_locs = df[col_name][changed.shift(-1, fill_value=True)].index
    return np.where(np.isin(df.index, boundary_locs))[0]


def raster_from_trains(
    trains,
    title=None,
    xlim=None,
    ax=None,
    structure_boundaries=True,
    probe_boundaries=True,
):
    MIN_UNITS_FOR_YTICKLABEL = 5

    if ax is None:
        fig, ax = plt.subplots(figsize=(36, len(trains) * 0.03))

    ax.eventplot(data=trains, positions="t", colors="rgba", linewidth=1)

    def _set_yticks(ax, col_name):
        boundary_ilocs = _get_boundary_ilocs(trains, col_name)
        ax.set_yticks(boundary_ilocs)
        do_label = np.diff(boundary_ilocs, prepend=0) > MIN_UNITS_FOR_YTICKLABEL
        ax.set_yticklabels(
            [
                trains[col_name].iloc[iloc] if label else ""
                for label, iloc in zip(do_label, boundary_ilocs)
            ]
        )

    if structure_boundaries and "structure" in trains.columns:
        _set_yticks(ax, "structure")
    else:
        ax.set_yticks([])

    if xlim is not None:
        ax.set_xlim(xlim)

    if probe_boundaries and "probe" in trains.columns:
        secy = ax.secondary_yaxis("right")
        _set_yticks(secy, "probe")
        ax.hlines(
            secy.get_yticks()[:-1],
            *ax.get_xlim(),
            color="red",
            alpha=0.8,
            linewidth=1,
            zorder=1
        )
        # n_probes = len(trains["probe"].unique())
        # probe_edges = np.insert(secy.get_yticks(), 0, 0)
        # probe_colors = (["whitesmoke", "lightgrey"] * n_probes)[:n_probes]
        # for lo, hi, c in zip(probe_edges, probe_edges[1:], probe_colors):
        #    ax.axhspan(lo, hi, color=c, alpha=0.1, ec=None, zorder=1)

    if title is not None:
        ax.set_title(title, loc="left")

    ax.margins(x=0, y=0)


def raster(
    sorting,
    plot_start,
    plot_length,
    ax,
):
    ax.cla()
    plot_end = plot_start + plot_length

    if set(["spikes", "units"]).issubset(sorting.keys()):
        trains = get_spike_trains_for_plotting(
            sorting.spikes, sorting.units, plot_start, plot_end
        )
    else:
        trains = get_multiprobe_spike_trains_for_plotting(sorting, plot_start, plot_end)
    raster_from_trains(trains, xlim=([plot_start, plot_end]), ax=ax)
    plt.tight_layout()


def raster_with_events(
    sorting,
    events,
    plot_start,
    plot_length,
    ax,
):
    raster(sorting, plot_start, plot_length, ax)
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
    def __init__(self, sorting):
        self._plotting_function = raster
        self._sorting = sorting
        self._plotting_args = {
            "sorting": fixed(self._sorting),
        }
        self.min_plot_duration = 1
        self.max_plot_duration = 60
        self.min_step_increment = 1
        self.setup_time_controls()

    @property
    def is_multiprobe(self):
        return not (set(["spikes", "units"]).issubset(self._sorting.keys()))

    @property
    def min_allowable_time(self):
        if self.is_multiprobe:
            return min(self._sorting[probe].spikes.t.min() for probe in self._sorting)
        else:
            return self._sorting.spikes.t.min()

    @property
    def max_allowable_time(self):
        if self.is_multiprobe:
            return max(self._sorting[probe].spikes.t.max() for probe in self._sorting)
        else:
            return self._sorting.spikes.t.max()
        # Techncially, we should subtract self.min_duration from this value, to prevent scrolling past the end of the data.

    @property
    def n_units(self):
        if self.is_multiprobe:
            return sum(len(self._sorting[probe].units) for probe in self._sorting)
        else:
            return len(self._sorting.units)

    def setup_time_controls(self):
        self._plot_start_slider = FloatSlider(
            min=self.min_allowable_time,
            max=self.max_allowable_time,
            step=self.min_step_increment,
            value=self.min_allowable_time,
            description="t=",
        )
        self._plot_start_box = BoundedFloatText(
            min=self.min_allowable_time,
            max=self.max_allowable_time,
            step=self.min_step_increment,
            value=self.min_allowable_time,
            description="t=",
        )
        jslink(
            (self._plot_start_slider, "value"), (self._plot_start_box, "value")
        )  # Allow control from either widget for easy navigation

        self._plot_length_box = BoundedFloatText(
            min=self.min_plot_duration,
            max=self.max_plot_duration,
            step=self.min_step_increment,
            value=np.ceil((self.max_plot_duration - self.min_plot_duration) / 2),
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

    def run(self, figsize="auto"):
        self.setup_ui()

        if figsize == "auto":
            figsize = (23, self.n_units * 0.03)
        elif figsize == "wide":
            figsize = (23, 8)
        elif figsize == "long":
            figsize = (9, 16)

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
    def __init__(self, sorting, events):
        super(InteractiveRasterWithEvents, self).__init__(sorting)
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
