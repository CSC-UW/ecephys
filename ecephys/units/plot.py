import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from IPython.display import display
from ipywidgets import (BoundedFloatText, BoundedIntText, FloatSlider, HBox,
                        Label, Layout, SelectMultiple, VBox, fixed,
                        interactive_output, jslink)
from matplotlib.colors import is_color_like, to_rgba
from matplotlib.ticker import IndexLocator

from ..acute import SHARPTrack

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


def _col_diff(df, col_name):
    return df[col_name].ne(df[col_name].shift().bfill())


def _get_boundary_ilocs(df, col_name):
    """Find trains.ilocs where trains['structure'] changes.
    These are the units that lie closest to structure boundaries, since `trains` is sorted by depth.

    Requres that df have an integer index!"""
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
    """Requires that df  have integer index"""
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

    ax.set_ylabel(f"Spikes grouped by {trains.index.name}, labelled by depth.")

    # Tick every _ rows with depth
    yticks = np.arange(0, len(trains), MIN_UNITS_FOR_YTICKLABEL).astype(int)
    yticklabels = trains.reset_index()['depth'][yticks].values
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)

    if structure_boundaries and "structure" in trains.columns:
        _set_yticks(ax, "structure")

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
            zorder=1,
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


class Raster:
    def __init__(
        self,
        sorting,
        plot_start=None,
        plot_duration=None,
        events=None,
        selection_levels=None,
        selections=None,
        grouping_col="cluster_id",
    ):
        self._sorting = sorting
        self._plot_start = plot_start
        self._plot_duration = plot_duration
        self._grouping_col=grouping_col
        self.update_trains()
        self.events = events
        if selection_levels is None:
            selection_levels = []
        self.selection_levels = selection_levels
        self.update_selection_options()
        if selections is None:
            selections = []
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
        if not set(val).issubset(
            self._trains.columns
        ):
            raise NotImplementedError(
                "Setting selection levels to any field in cluster_info.tsv is not yet supported "
                "when grouping spikes by another column than 'cluster_id'. "
                "The supported selection levels for this type of Sorting/grouping are: "
                f"`{self._trains.columns}`. \n"
                "Modify `SingleProbeSorting.get_spike_trains_for_plotting` to support all levels from cluster_info.tsv."
            )
        # assert set(val).issubset(
        #     self._trains.columns
        # ), f"val={val}, cols={self._trains.columns}"
        self._selection_levels = list(val)

    def update_selection_options(self):
        if self.selection_levels:
            self._selection_options = (
                self._trains.set_index(self.selection_levels)
                .index.to_flat_index()
                .unique()
                .to_list()
            )
        else:
            self._selection_options = []

    @property
    def selection_options(self):
        return self._selection_options

    @property
    def selections(self):
        return self._selections

    @selections.setter
    def selections(self, val):
        assert set(val).issubset(
            self.selection_options
        ), f"val={val}, selection_options={self.selection_options}"
        self._selections = list(val)

    @property
    def trains(self):
        if self.selection_levels:
            trains = self._trains.set_index(self.selection_levels, drop=False)
            trains.index = trains.index.to_flat_index()
            return trains.drop(self.selections).reset_index(drop=True)
        else:
            return self._trains

    def update_trains(self):
        self._trains = self._sorting.get_spike_trains_for_plotting(
            start_time=self.plot_start,
            end_time=self.plot_end,
            grouping_col=self._grouping_col,
        )

    def plot(self, figsize="auto"):
        figsize = self.figsizes.get(figsize, figsize)
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
        ) | (
            (self.events["t1"] <= self.plot_start)
            & (self.events["t2"] >= self.plot_end)
        )

        for evt in self.events[mask].itertuples():
            ax.axvspan(
                max(evt.t1, self.plot_start),
                min(evt.t2, self.plot_end),
                fc=to_rgba("lavender", 0.3),
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
            layout=Layout(width="95%"),
        )
        plot_start_box = BoundedFloatText(
            min=self.min_plot_start,
            max=self.max_plot_start,
            step=self.min_plot_duration,
            value=self.plot_start,
            description="t=",
            layout=Layout(width="150px"),
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
            layout=Layout(width="150px"),
        )
        # Slider step equal to plot duration for easier scrolling
        jslink(
            (plot_duration_box, "value"), (plot_start_box, "step")
        )
        jslink(
            (plot_start_box, "step"), (plot_start_slider, "step")
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
            layout=Layout(width="150px"),
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
        figsize = self.figsizes.get(figsize, figsize)
        fig, ax = plt.subplots(figsize=figsize)
        fig.canvas.header_visible = False
        fig.canvas.toolbar_visible = False

        [
            plot_start_slider,
            plot_start_box,
            plot_duration_box,
        ] = self.get_time_controls()

        if self.events is not None:
            event_controls = self.get_event_controls(plot_start_box, plot_duration_box)
            navigation = VBox(
                [
                    plot_start_slider,
                    HBox([plot_start_box, plot_duration_box] + event_controls),
                ]
            )
        else:
            navigation = VBox(
                [plot_start_slider, HBox([plot_start_box, plot_duration_box])]
            )

        if self.selection_levels:
            menu = SelectMultiple(
                options=[(str(opt), opt) for opt in reversed(self.selection_options)],
                value=self.selections,
                description="Hide:",
                rows=min(10, len(self.selection_options)),
                disabled=False,
            )
            ui = VBox([menu, navigation])
        else:
            menu = fixed([])
            ui = navigation

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
