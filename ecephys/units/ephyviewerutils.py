import ephyviewer
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.colors


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
    sorting,
    by="cluster_id",
    tgt_struct_acronym=None,
    probe=None,
    view_params=None,
):

    properties = sorting.properties

    if view_params is None:
        view_params = {"display_labels": False}

    # Structure-wide information
    if tgt_struct_acronym is not None:
        mask = properties["acronym"] == tgt_struct_acronym
        tgt_properties = properties[mask]
        lo = sorting.structs.set_index("acronym").loc[tgt_struct_acronym, "lo"]
        hi = sorting.structs.set_index("acronym").loc[tgt_struct_acronym, "hi"]
        view_name = f"Structure: {tgt_struct_acronym}, Depths: {lo}-{hi}um"
    else:
        tgt_properties = properties
        lo = sorting.structs.lo.min()
        hi = sorting.structs.hi.max()
        view_name = f"Struct: full probe, {lo}-{hi}um"
    view_name = f"{view_name}, N={len(tgt_properties)} units"
    if probe is not None:
        view_name = f"Probe: {probe}, {view_name}"

    # Get tgt values for grouping trains within structure
    if by == "depth":
        # Descending depths between structure min/max in 20um steps
        tgt_values = np.arange(lo, hi + DEPTH_STEP, DEPTH_STEP)[::-1]
    else:
        # Units sorted by depth within each structure
        sorted_tgt_properties = tgt_properties.sort_values(by="depth", ascending=False)
        tgt_values = sorted_tgt_properties[by].unique()  # Still sorted

    all_struct_spikes = []
    for tgt_value in tqdm(tgt_values, desc=f"Loading spikes for view: `{view_name}`"):

        label = f"{by}: {tgt_value}"

        if by != "cluster_id":
            ids = tgt_properties[tgt_properties[by] == tgt_value].cluster_id.values
            label = f"{label}, ids={ids}"

        all_struct_spikes.append(
            {
                "time": sorting.get_trains(
                    by=by,
                    tgt_values=[tgt_value],
                    verbose=False,
                )[tgt_value],
                "name": label,
            }
        )

    source = ephyviewer.InMemorySpikeSource(all_spikes=all_struct_spikes)
    view = ephyviewer.SpikeTrainViewer(source=source, name=view_name)

    for p, v in view_params.items():
        view.params[p] = v

    window.add_view(view, location="bottom", orientation="vertical")

    return window
