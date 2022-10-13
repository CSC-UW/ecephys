import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from IPython.display import display
from ipywidgets import (
    BoundedFloatText,
    BoundedIntText,
    FloatSlider,
    HBox,
    Label,
    Layout,
    SelectMultiple,
    VBox,
    fixed,
    interactive_output,
    jslink,
)
from matplotlib.colors import is_color_like, to_rgba
from ecephys.units import sorting, spikes

from ..sharptrack import SHARPTrack

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


def _set_yticks_at_boundaries(
    ax, trns, boundariesBetween="structure", minSizeRequiredForLabel=5
):
    # Plot YTicks that mark boundaries between levels of a property, e.g. structure or probe.
    ilocs = _get_boundary_ilocs(trns, boundariesBetween)
    ax.set_yticks(ilocs)
    do_label = np.diff(ilocs, prepend=0) > minSizeRequiredForLabel
    ax.set_yticklabels(
        [
            trns[boundariesBetween].iloc[iloc] if label else ""
            for label, iloc in zip(do_label, ilocs)
        ]
    )


def raster_from_trains(
    trns,
    title=None,
    xlim=None,
    ax=None,
    structure_boundaries=True,
    probe_boundaries=True,
    spikesize=1,
):
    """Requires that df  have integer index"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(36, len(trns) * 0.03))

    if "rgba" not in trns.columns:
        set_uniform_rgba(trns, "black")

    ax.eventplot(data=trns, positions="t", colors="rgba", linewidth=spikesize)
    ax.set_ylabel(f"One train per {trns.index.name}")

    minYLabelSpacing = 5
    if structure_boundaries and "structure" in trns.columns:
        _set_yticks_at_boundaries(ax, trns, "structure", minYLabelSpacing)
    elif (
        trns.index.name == "depth"
    ):  # This is ugly. Find a better way, or separate funtions for depth-indexed trains and clusters-indexed trains.
        yticks = np.arange(0, len(trns), minYLabelSpacing).astype(int)
        yticklabels = trns.reset_index()["depth"][yticks].values
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)

    if xlim is not None:
        ax.set_xlim(xlim)

    if probe_boundaries and "probe" in trns.columns:
        secy = ax.secondary_yaxis("right")
        _set_yticks_at_boundaries(secy, trns, "probe", minYLabelSpacing)
        ax.hlines(
            secy.get_yticks()[:-1],
            *ax.get_xlim(),
            color="red",
            alpha=0.8,
            linewidth=1,
            zorder=1,
        )

    if title is not None:
        ax.set_title(title, loc="left")

    ax.margins(x=0, y=0)
    plt.tight_layout()

    return ax


class Raster:
    def __init__(
        self,
        sorting,
        plotStart=None,
        plotDuration=None,
        events=None,
        selectionLevels=None,
        selections=None,
        oneTrainPer="cluster_id",
        alpha=0.3,
    ):
        """
        Parameters:
        ===========
        sorting: units.Sorting
            NOT a spikeinterface sorting object
        """
        self._sorting = sorting
        self._plotStart = plotStart
        self._plotDuration = plotDuration
        self._oneTrainPer = oneTrainPer
        self.update_trains()
        self.events = events
        if selectionLevels is None:
            selectionLevels = []
        self.selectionLevels = selectionLevels
        self.update_selection_options()
        if selections is None:
            selections = []
        self.selections = selections
        self.alpha = alpha

        self.figsizes = {
            "auto": (23, self._sorting.nClusters * 0.03),
            "wide": (23, 8),
            "long": (9, 16),
        }

    @property
    def firstSpikeTime(self):
        return self._sorting.firstSpikeTime

    @property
    def lastSpikeTime(self):
        return self._sorting.lastSpikeTime

    @property
    def minPlotDuration(self):
        return min(0.1, self.lastSpikeTime - self.firstSpikeTime)  # Try 100ms

    @property
    def maxPlotDuration(self):
        return self.lastSpikeTime - self.firstSpikeTime

    @property
    def minPlotStart(self):
        return np.floor(self.firstSpikeTime)

    @property
    def maxPlotStart(self):
        return self.lastSpikeTime - self.minPlotDuration

    @property
    def plotStart(self):
        if self._plotStart is None:
            self._plotStart = self.minPlotStart
        return self._plotStart

    @plotStart.setter
    def plotStart(self, val):
        if val < self.minPlotStart:
            self._plotStart = self.minPlotStart
        elif val >= self.maxPlotStart:
            self._plotStart = self.maxPlotStart
        else:
            self._plotStart = val

        self.update_trains()

    @property
    def plotDuration(self):
        if self._plotDuration is None:
            self._plotDuration = self.minPlotDuration
        return self._plotDuration

    @plotDuration.setter
    def plotDuration(self, val):
        if val > self.maxPlotDuration:
            self._plotDuration = self.maxPlotDuration
        elif val <= self.minPlotDuration:
            self._plotDuration = self.minPlotDuration
        else:
            self._plotDuration = val

        self.update_trains()

    @property
    def plotEnd(self):
        return self.plotStart + self.plotDuration

    @property
    def selectionLevels(self):
        return self._selectionLevels

    @selectionLevels.setter
    def selectionLevels(self, val):
        assert set(val).issubset(
            self._trains.columns
        ), f"val={val}, cols={self._trains.columns}"
        self._selectionLevels = list(val)

    # TODO: This will no longer work, unless trains already include selection levels
    # TODO: Separate ClusterRaster and DepthRaster?
    # Most of the complexity in this class comes from the selection levels, which are rarely used.
    def update_selection_options(self):
        if self.selectionLevels:
            self._selectionOptions = (
                self._trains.set_index(self.selectionLevels)
                .index.to_flat_index()
                .unique()
                .to_list()
            )
        else:
            self._selectionOptions = []

    @property
    def selectionOptions(self):
        return self._selectionOptions

    @property
    def selections(self):
        return self._selections

    @selections.setter
    def selections(self, val):
        assert set(val).issubset(
            self.selectionOptions
        ), f"val={val}, selection_options={self.selectionOptions}"
        self._selections = list(val)

    @property
    def trains(self):
        if self.selectionLevels:
            trns = self._trains.set_index(self.selectionLevels, drop=False)
            trns.index = trns.index.to_flat_index()
            return trns.drop(self.selections).reset_index(drop=True)
        else:
            return self._trains

    def update_trains(self):
        if isinstance(self._sorting, sorting.SingleProbeSorting):
            # Get spikes between the requested times
            spks = spikes.between_time(
                self._sorting.spikes, self.plotStart, self.plotEnd
            )
            # If spike trains should be grouped by something other than cluster_id, we need to add that info to the spikes frame.
            if self._oneTrainPer != "cluster_id":
                spks = spikes.add_cluster_info(
                    spks, self._sorting.clusterInfo, self._oneTrainPer
                )
            # Get the spike trains
            self._trains = spikes.as_trains(spks, self._oneTrainPer)
        else:
            raise NotImplementedError("MultiProbeSorting not yet implemented")

    def plot(self, figsize="auto"):
        figsize = self.figsizes.get(figsize, figsize)
        fig, ax = plt.subplots(figsize=figsize)

        fig.canvas.header_visible = False
        fig.canvas.toolbar_visible = False
        ax = raster_from_trains(self.trains, xlim=[self.plotStart, self.plotEnd], ax=ax)
        if self.events is not None:
            self.add_event_overlay(ax)
        return ax

    def add_event_overlay(self, ax):
        mask = (
            (
                (self.events["t1"] >= self.plotStart)
                & (self.events["t1"] <= self.plotEnd)
            )
            | (
                (self.events["t2"] >= self.plotStart)
                & (self.events["t2"] <= self.plotEnd)
            )
            | (
                (self.events["t1"] <= self.plotStart)
                & (self.events["t2"] >= self.plotEnd)
            )
        )

        AXVSPAN_KWARGS = {
            "fc": to_rgba("darkred", 0.1),
            "ec": to_rgba("darkred", 1.0),
            "linewidth": 1,
            "alpha": self.alpha,
        }

        for _, evt_row in self.events[mask].iterrows():
            if "ylim" in evt_row:
                ymin, ymax = evt_row.ylim
                plt.margins(0)  # Remove margin on y axis
            else:
                ymin, ymax = 0, 1
            kwargs = {col: evt_row.get(col, df) for col, df in AXVSPAN_KWARGS.items()}
            ax.axvspan(
                max(evt_row.t1, self.plotStart),
                min(evt_row.t2, self.plotEnd),
                ymin=ymin,
                ymax=ymax,
                **kwargs,
            )

    def _interact(self, plot_start, plot_duration, selections, ax):
        ax.cla()
        self.plotStart = plot_start
        self.plotDuration = plot_duration
        self.selections = selections
        ax = raster_from_trains(self.trains, xlim=[self.plotStart, self.plotEnd], ax=ax)
        if self.events is not None:
            self.add_event_overlay(ax)
        return ax

    def get_time_controls(self):
        plotStartSlider = FloatSlider(
            min=self.minPlotStart,
            max=self.maxPlotStart,
            step=self.minPlotDuration,
            value=self.plotStart,
            description="t=",
            continuous_update=False,
            layout=Layout(width="95%"),
        )
        plotStartBox = BoundedFloatText(
            min=self.minPlotStart,
            max=self.maxPlotStart,
            step=self.minPlotDuration,
            value=self.plotStart,
            description="t=",
            layout=Layout(width="150px"),
        )
        jslink(
            (plotStartSlider, "value"), (plotStartBox, "value")
        )  # Allow control from either widget for easy navigation

        plotDurationBox = BoundedFloatText(
            min=self.minPlotDuration,
            max=self.maxPlotDuration,
            step=self.minPlotDuration,
            value=self.plotDuration,
            description="Secs",
            layout=Layout(width="150px"),
        )
        # Slider step equal to plot duration for easier scrolling
        jslink((plotDurationBox, "value"), (plotStartBox, "step"))
        jslink((plotStartBox, "step"), (plotStartSlider, "step"))

        return [plotStartSlider, plotStartBox, plotDurationBox]

    def get_event_controls(self, plotStartBox, plotDurationBox):
        assert self.events.index.dtype == "int64"
        if "t2" not in self.events.columns:
            self.events = self.events.copy()
            self.events["t2"] = self.events["t1"]

        eventBox = BoundedIntText(
            value=self.events.index.min(),
            min=self.events.index.min(),
            max=self.events.index.max(),
            step=1,
            description="EVT",
            layout=Layout(width="150px"),
        )
        eventDescriptionLabel = Label(value=self.events.iloc[0]["description"])

        def _on_event_change(change):
            evt = change["new"]
            t1 = self.events.loc[evt, "t1"]
            desc = self.events.loc[evt, "description"]
            eventDescriptionLabel.value = desc
            plotStartBox.value = t1 - plotDurationBox.value / 2

        eventBox.observe(_on_event_change, names="value")

        return [eventBox, eventDescriptionLabel]

    def interact(self, figsize="auto"):
        figsize = self.figsizes.get(figsize, figsize)
        fig, ax = plt.subplots(figsize=figsize)
        fig.canvas.header_visible = False
        fig.canvas.toolbar_visible = False

        [
            plotStartSlider,
            plotStartBox,
            plotDurationBox,
        ] = self.get_time_controls()

        if self.events is not None:
            eventControls = self.get_event_controls(plotStartBox, plotDurationBox)
            navigation = VBox(
                [
                    plotStartSlider,
                    HBox([plotStartBox, plotDurationBox] + eventControls),
                ]
            )
        else:
            navigation = VBox([plotStartSlider, HBox([plotStartBox, plotDurationBox])])

        if self.selectionLevels:
            menu = SelectMultiple(
                options=[(str(opt), opt) for opt in reversed(self.selectionOptions)],
                value=self.selections,
                description="Hide:",
                rows=min(10, len(self.selectionOptions)),
                disabled=False,
            )
            ui = VBox([menu, navigation])
        else:
            menu = fixed([])
            ui = navigation

        out = interactive_output(
            self._interact,
            {
                "plot_start": plotStartBox,
                "plot_duration": plotDurationBox,
                "selections": menu,
                "ax": fixed(ax),
            },
        )
        display(ui, out)
        # When using %matplotlib widget backend, it is suggested that we do not use out = interactive_output(...); display(ui, out)
        # Instead, it is suggested that we use display(ui, fig.canvas), or include fig.canvas in the ui,
        # because interactive_output(...) should be reserved for an inline backend.
        # See: https://github.com/matplotlib/matplotlib/issues/23229
        # One would assume that it would also be possible to not use the %matplotlib widget backend and keep interactive_output(...),
        # possibly with a plt.ion() call before the Raster object is created. But this does not work.
        # It seems that ipywigets.AppLayout is the recommended way to not use the %matplotlib widget backend.
        # See: https://ipywidgets.readthedocs.io/en/latest/examples/Layout%20Templates.html
