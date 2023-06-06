import ephyviewer
import matplotlib.colors
import numpy as np
import pandas as pd
from tqdm import tqdm

from ecephys.units import SpikeInterfaceKilosortSorting


DEPTH_STEP = 20


def add_traceviewer_to_window(
    window: ephyviewer.MainViewer,
    sigs: np.array,
    sample_rate: float,
    t_start: float,
    channel_names: np.array,
    view_name: str = "Traces",
    view_params: dict = None,
):
    """Add traceviewer

    Parameters:
    window: ephyviewer.MainWindow
    sigs: np.array
        (nchans x nsamples)
    sample_rate
    t_start
    """
    source = ephyviewer.InMemoryAnalogSignalSource(
        np.transpose(sigs), sample_rate, t_start, channel_names=channel_names
    )

    view = ephyviewer.TraceViewer(source=source, name=view_name)

    if view_params is None:
        view_params = {}
    for p, v in view_params.items():
        print(p, v)
        view.params[p] = v

    window.add_view(view, location="bottom")

    return window


def add_epochviewer_to_window(
    window: ephyviewer.MainViewer,
    events_df: pd.DataFrame,
    view_name="Events",
    name_column="state",
    color_by_name=None,
    add_event_list=True,
):
    """Add epoch/event view from frame of bouts.

    Parameters:
    window: ephyviewer.MainViewer
    event_df: pd.DataFrame
        Frame with "start_time", "duration" and <name_column> columns
    """
    assert all([c in events_df.columns for c in ["start_time", "state", "duration"]])

    all_names = events_df[name_column].unique()
    all_epochs = []

    for name in all_names:
        mask = events_df[name_column] == name
        all_epochs.append(
            {
                "name": name,
                "label": np.array([name_column for _ in range(mask.sum())]),
                "time": events_df[mask]["start_time"].values,
                "duration": events_df[mask]["duration"].values,
            }
        )

    source_epochs = ephyviewer.InMemoryEpochSource(all_epochs=all_epochs)
    view = ephyviewer.EpochViewer(source=source_epochs, name=view_name)

    # Set usual colors
    if color_by_name:
        for i, name in enumerate(all_names):
            view.by_channel_params[f"ch{i}", "color"] = matplotlib.colors.rgb2hex(
                color_by_name[name]
            )

    window.add_view(view, location="bottom", orientation="vertical")

    if add_event_list:
        # Add event list for navigation
        view = ephyviewer.EventList(source=source_epochs, name=f"{view_name} list")
        window.add_view(view, orientation="horizontal", split_with=view_name)

    return window


def add_spiketrainviewer_to_window(
    window: ephyviewer.MainViewer,
    sorting: SpikeInterfaceKilosortSorting,
    by="cluster_id",
    probe=None,
    view_params=None,
):
    """Add a single panel for all units, sorted by depth, grouped by <by>."""

    properties = sorting.properties

    if view_params is None:
        view_params = {"display_labels": False}

    # Structure-wide information
    lo = sorting.structs.lo.values
    hi = sorting.structs.hi.values
    view_name = f"Structures: {sorting.structures_by_depth}, lo={lo}-{hi}um, N={len(properties)}"
    if probe is not None:
        view_name = f"Probe: {probe}, {view_name}"

    # Get tgt values for grouping trains (We want to represent empty depths)
    if by == "depth":
        # Descending depths between structure min/max in 20um steps
        min_val, max_val = lo.min(), hi.max()
        tgt_values = np.arange(min_val, max_val + DEPTH_STEP, DEPTH_STEP)[::-1]
    else:
        # Units sorted by depth within each structure
        sorted_tgt_properties = properties.sort_values(by="depth", ascending=False)
        tgt_values = sorted_tgt_properties[by].unique()  # Still sorted

    all_trains = []
    for tgt_value in tqdm(tgt_values, desc=f"Loading spikes for view: `{view_name}`"):
        label = f"{by}: {tgt_value}"

        if by != "cluster_id":
            ids = properties[properties[by] == tgt_value].cluster_id.values
            label = f"{label}, ids={ids}"

        all_trains.append(
            {
                "time": sorting.get_trains_by_property(
                    property_name=by,
                    values=[tgt_value],
                    verbose=False,
                )[tgt_value],
                "name": label,
            }
        )

    source = ephyviewer.InMemorySpikeSource(all_spikes=all_trains)
    view = ephyviewer.SpikeTrainViewer(source=source, name=view_name)

    for p, v in view_params.items():
        view.params[p] = v

    window.add_view(view, location="bottom", orientation="vertical")

    return window


def add_spatialoff_viewer_to_window(
    window: ephyviewer.MainViewer,
    off_df: pd.DataFrame,
    view_name="Spatial offs",
    t1_column="start_time",
    t2_column="end_time",
    d1_column="depth_min",
    d2_column="depth_max",
    ylim=None,
    Tmax=None,
    binsize=0.01,
    add_event_list=True,
):
    """Add spatial off viewer (from TraceImageViewer)"""
    assert all(
        [c in off_df.columns for c in [t1_column, t2_column, d1_column, d2_column]]
    )

    if ylim is None:
        ylim = (off_df[d1_column].min(), off_df[d2_column].max())
    if Tmax is None:
        Tmax = off_df[t2_column].max()

    timestamps = np.arange(0, Tmax, binsize)
    depthstamps = np.arange(ylim[0], ylim[1] + 10, 10)
    off_image = np.zeros((len(depthstamps), len(timestamps)), dtype=float)

    for row in tqdm(list(off_df.itertuples())):
        t1_idx = np.searchsorted(timestamps, getattr(row, t1_column))
        t2_idx = np.searchsorted(timestamps, getattr(row, t2_column))
        d1_idx = np.searchsorted(depthstamps, getattr(row, d1_column))
        d2_idx = np.searchsorted(depthstamps, getattr(row, d2_column))
        off_image[d1_idx:d2_idx, t1_idx:t2_idx] = 1

    source = ephyviewer.InMemoryAnalogSignalSource(
        np.transpose(off_image),
        1 / binsize,
        0,
        channel_names=depthstamps,
    )

    view = ephyviewer.TraceImageViewer(source=source, name=view_name)
    window.add_view(view, location="bottom")

    if add_event_list:
        # Add event list for navigation
        epochs = []
        labels = (
            off_df[d1_column].astype(str) + "–" + off_df[d2_column].astype(str) + "um"
        ).values
        epochs.append(
            {
                "name": "offs",
                "label": labels,
                "time": off_df[t1_column].values,
                "duration": (off_df[t2_column] - off_df[t1_column]).values,
            }
        )

        source_epochs = ephyviewer.InMemoryEpochSource(all_epochs=epochs)

        view = ephyviewer.EventList(source=source_epochs, name=f"{view_name} list")
        window.add_view(view, orientation="horizontal", split_with=view_name)

    return window
