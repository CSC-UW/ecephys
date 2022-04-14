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
    SelectMultiple,
    HBox,
    fixed,
    interactive_output,
    jslink,
)
from abc import ABC, abstractproperty, abstractstaticmethod, abstractmethod
import warnings

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


# Remove, and use as SingleProbeSorting classmethod?
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


# Remove, and use as MultiProbeSorting classmethod?
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
    These are the units that lie closest to structure boundaries, since `trains` is sorted by depth.

    Requres that df have an integer index!"""
    df = df.reset_index()
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
    plt.tight_layout()

    return ax


# Remove?
def raster(
    sorting,
    plot_start,
    plot_length,
    ax,
):
    ax.cla()
    plot_end = plot_start + plot_length

    trains = sorting.plotting_trains(plot_start, plot_end)
    raster_from_trains(trains, xlim=([plot_start, plot_end]), ax=ax)
    plt.tight_layout()


# Remove?
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


# Remove?
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


# Remove?
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


##### DEV #####


class Sorting(ABC):
    def __init__(self, bunch):
        self._bunch = bunch

    @abstractproperty
    def data_start(self):
        pass

    @abstractproperty
    def data_end(self):
        pass

    @abstractproperty
    def n_units(self):
        pass

    @abstractmethod
    def get_spike_trains_for_plotting(self, start_time, end_time):
        pass


class SingleProbeSorting(Sorting):
    def __init__(self, bunch):
        super(SingleProbeSorting, self).__init__(bunch)

    @property
    def data_start(self):
        return self._bunch.spikes.t.min()

    @property
    def data_end(self):
        return self._bunch.spikes.t.max()
        # Techncially, we should subtract self.min_duration from this value, to prevent scrolling past the end of the data.

    @property
    def n_units(self):
        return len(self._bunch.units)

    def get_spike_trains_for_plotting(self, start_time=None, end_time=None):
        start_time = self.data_start if start_time is None else start_time
        end_time = self.data_end if end_time is None else end_time

        trains = self._bunch.spikes.spikes.as_trains(start_time, end_time)

        if "rgba" not in self._bunch.units.columns:
            set_uniform_rgba(self._bunch.units, "black")

        trains = self._bunch.units.join(trains, how="outer")
        silent = trains.trains.silent()
        # Add ghost spikes at very start and end of window to silent trains, to reserve space for them on the plot's x and y axes.
        trains.loc[silent, "t"] = pd.Series(
            [np.array((start_time, end_time))] * sum(silent)
        ).values
        # Make silent units white and transparent, so that they are invisible.
        trains.loc[silent, "rgba"] = pd.Series(
            [to_rgba("white", 0.0)] * sum(silent)
        ).values
        return trains.sort_values("depth")


class MultiProbeSorting(Sorting):
    def __init__(self, bunch):
        super(MultiProbeSorting, self).__init__(bunch)

    @property
    def data_start(self):
        return min(self._bunch[probe].spikes.t.min() for probe in self._bunch)

    @property
    def data_end(self):
        return max(self._bunch[probe].spikes.t.max() for probe in self._bunch)
        # Techncially, we should subtract self.min_duration from this value, to prevent scrolling past the end of the data.

    @property
    def n_units(self):
        return sum(len(self._bunch[probe].units) for probe in self._bunch)

    def get_spike_trains_for_plotting(self, start_time=None, end_time=None):
        start_time = self.data_start if start_time is None else start_time
        end_time = self.data_end if end_time is None else end_time
        multiprobe_bunch = self._bunch

        trains = Bunch()
        for probe in multiprobe_bunch:
            singleprobe_sorting = SingleProbeSorting(multiprobe_bunch[probe])
            trains[probe] = singleprobe_sorting.get_spike_trains_for_plotting(
                start_time,
                end_time,
            )

        return pd.concat(
            [trains[probe] for probe in trains], keys=trains.keys(), names=["probe"]
        ).reset_index()


class Raster:
    def __init__(
        self,
        sorting,
        plot_start=None,
        plot_duration=None,
        events=None,
        selection_levels=[],
        selections=[],
    ):
        self._sorting = sorting
        self._plot_start = plot_start
        self._plot_duration = plot_duration
        self.update_trains()
        self.events = events
        self.selection_levels = selection_levels
        self.update_selection_options()
        self.selections = selections

        self.figsizes = {
            "auto": (23, self._sorting.n_units * 0.03),
            "wide": (23, 8),
            "long": (9, 16),
        }

    @property
    def data_start(self):
        return self._sorting.data_start

    @property
    def data_end(self):
        return self._sorting.data_end

    @property
    def min_plot_duration(self):
        return min(0.1, self.data_end - self.data_start)  # Try 100ms

    @property
    def max_plot_duration(self):
        return self.data_end - self.data_start

    @property
    def min_plot_start(self):
        return np.floor(self.data_start)

    @property
    def max_plot_start(self):
        return self.data_end - self.min_plot_duration

    @property
    def plot_start(self):
        if self._plot_start is None:
            self._plot_start = self.min_plot_start
        return self._plot_start

    @plot_start.setter
    def plot_start(self, val):
        if val < self.min_plot_start:
            self._plot_start = self.min_plot_start
        elif val >= self.max_plot_start:
            self._plot_start = self.max_plot_start
        else:
            self._plot_start = val

        self.update_trains()

    @property
    def plot_duration(self):
        if self._plot_duration is None:
            self._plot_duration = self.min_plot_duration
        return self._plot_duration

    @plot_duration.setter
    def plot_duration(self, val):
        if val > self.max_plot_duration:
            self._plot_duration = self.max_plot_duration
        elif val <= self.min_plot_duration:
            self._plot_duration = self.min_plot_duration
        else:
            self._plot_duration = val

        self.update_trains()

    @property
    def plot_end(self):
        return self.plot_start + self.plot_duration

    @property
    def selection_levels(self):
        return self._selection_levels

    @selection_levels.setter
    def selection_levels(self, val):
        assert set(val).issubset(self._trains.columns)
        self._selection_levels = list(val)

    def update_selection_options(self):
        self._selection_options = (
            self._trains.set_index(self.selection_levels)
            .index.to_flat_index()
            .unique()
            .to_list()
        )

    @property
    def selection_options(self):
        return self._selection_options

    @property
    def selections(self):
        return self._selections

    @selections.setter
    def selections(self, val):
        assert set(val).issubset(self.selection_options)
        self._selections = list(val)

    @property
    def trains(self):
        if (self.selection_levels) and (self.selections):
            trains = self._trains.set_index(self.selection_levels, drop=False)
            trains.index = trains.index.to_flat_index()
            return trains.drop(self.selections)
        else:
            return self._trains

    def update_trains(self):
        self._trains = self._sorting.get_spike_trains_for_plotting(
            self.plot_start, self.plot_end
        )

    def plot(self, figsize="auto"):
        figsize = self.figsizes.pop(figsize, figsize)
        fig, ax = plt.subplots(figsize=figsize)

        fig.canvas.header_visible = False
        fig.canvas.toolbar_visible = False
        ax = raster_from_trains(
            self.trains, xlim=[self.plot_start, self.plot_end], ax=ax
        )
        if self.events is not None:
            self.add_event_overlay(ax)
        return ax

    def add_event_overlay(self, ax):
        mask = (
            (self.events["t1"] >= self.plot_start)
            & (self.events["t1"] <= self.plot_end)
        ) | (
            (self.events["t2"] >= self.plot_start)
            & (self.events["t2"] <= self.plot_end)
        )

        for evt in self.events[mask].itertuples():
            ax.axvspan(
                max(evt.t1, self.plot_start),
                min(evt.t2, self.plot_end),
                fc=to_rgba("lavender", 0.1),
                ec=to_rgba("lavender", 1.0),
            )

    def _interact(self, plot_start, plot_duration, selections, ax):
        ax.cla()
        self.plot_start = plot_start
        self.plot_duration = plot_duration
        self.selections = selections
        ax = raster_from_trains(
            self.trains, xlim=[self.plot_start, self.plot_end], ax=ax
        )
        if self.events is not None:
            self.add_event_overlay(ax)
        return ax

    def get_time_controls(self):
        plot_start_slider = FloatSlider(
            min=self.min_plot_start,
            max=self.max_plot_start,
            step=self.min_plot_duration,
            value=self.plot_start,
            description="t=",
            continuous_update=False,
        )
        plot_start_box = BoundedFloatText(
            min=self.min_plot_start,
            max=self.max_plot_start,
            step=self.min_plot_duration,
            value=self.plot_start,
            description="t=",
        )
        jslink(
            (plot_start_slider, "value"), (plot_start_box, "value")
        )  # Allow control from either widget for easy navigation

        plot_duration_box = BoundedFloatText(
            min=self.min_plot_duration,
            max=self.max_plot_duration,
            step=self.min_plot_duration,
            value=self.plot_duration,
            description="Secs",
        )

        return [plot_start_slider, plot_start_box, plot_duration_box]

    def get_event_controls(self, plot_start_box, plot_duration_box):
        assert self.events.index.dtype == "int64"
        if "t2" not in self.events.columns:
            self.events = self.events.copy()
            self.events["t2"] = self.events["t1"]

        event_box = BoundedIntText(
            value=self.events.index.min(),
            min=self.events.index.min(),
            max=self.events.index.max(),
            step=1,
            description="EVT",
        )
        event_description_label = Label(value=self.events.iloc[0]["description"])

        def _on_event_change(change):
            evt = change["new"]
            t1 = self.events.loc[evt, "t1"]
            desc = self.events.loc[evt, "description"]
            event_description_label.value = desc
            plot_start_box.value = t1 - plot_duration_box.value / 2

        event_box.observe(_on_event_change, names="value")

        return [event_box, event_description_label]

    def interact(self, figsize="auto"):
        figsize = self.figsizes.pop(figsize, figsize)
        fig, ax = plt.subplots(figsize=figsize)
        fig.canvas.header_visible = False
        fig.canvas.toolbar_visible = False

        [
            plot_start_slider,
            plot_start_box,
            plot_duration_box,
        ] = controls = self.get_time_controls()

        if self.events is not None:
            controls = controls + self.get_event_controls(
                plot_start_box, plot_duration_box
            )

        if self.selection_levels:
            menu = SelectMultiple(
                options=[(str(opt), opt) for opt in self.selection_options],
                value=self.selections,
                rows=3,
                description="Hide:",
                disabled=False,
            )
            controls = controls + [menu]
        else:
            menu = fixed([])

        ui = HBox(controls)
        out = interactive_output(
            self._interact,
            {
                "plot_start": plot_start_box,
                "plot_duration": plot_duration_box,
                "selections": menu,
                "ax": fixed(ax),
            },
        )
        display(ui, out)
