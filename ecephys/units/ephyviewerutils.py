import ephyviewer
from tqdm import tqdm
import numpy as np
from ecephys.plot import state_colors
import pandas as pd
import matplotlib.colors


DEPTH_STEP = 20


def add_hypnogram_to_window(window: ephyviewer.MainViewer, hypnogram: pd.DataFrame):

    all_states = hypnogram.state.unique()
    all_epochs = []

    for state in all_states:
        mask = hypnogram["state"] == state
        all_epochs.append({
            "name": state,
            "label": np.array([state for _ in range(mask.sum())]),
            "time": hypnogram[mask].start_time.values,
            "duration": hypnogram[mask].duration.values,
        })

    source_epochs = ephyviewer.InMemoryEpochSource(all_epochs=all_epochs)
    view = ephyviewer.EpochViewer(source=source_epochs, name='Hypnogram')

    # Set usual Hypnogram colors
    for i, state in enumerate(all_states):
        view.by_channel_params[f"ch{i}", 'color'] = matplotlib.colors.rgb2hex(state_colors[state])

    window.add_view(view, location="bottom", orientation="vertical")

    # Add event list for navigation
    view = ephyviewer.EventList(source=source_epochs, name='event')
    window.add_view(view, location='bottom',  orientation='horizontal')

    return window


def add_spiketrainviewer_to_window(
        window: ephyviewer.MainViewer, sorting, by="cluster_id", tgt_struct_acronym=None, probe=None,
    ):

    properties = sorting.properties
    
    # Structure-wide information
    if tgt_struct_acronym is not None:
        mask = properties["acronym"] == tgt_struct_acronym
        tgt_properties = properties[mask]
        lo = sorting.structs.set_index("acronym").loc[tgt_struct_acronym, "lo"]
        hi = sorting.structs.set_index("acronym").loc[tgt_struct_acronym, "hi"]
        view_name = f"Struct: {tgt_struct_acronym}, {lo}-{hi}um"
    else:
        tgt_properties = properties
        lo = sorting.structs.lo.min()
        hi = sorting.structs.hi.max()
        view_name = "Struct: full probe, {lo}-{hi}um"
    if probe is not None:
        view_name += f", probe={probe}"

    # Get tgt values for grouping trains within structure
    if by == "depth":
        # Descending depths between structure min/max in 20um steps
        tgt_values = np.arange(lo, hi + DEPTH_STEP, DEPTH_STEP)[::-1]
    else:
        # Units sorted by depth within each structure
        sorted_tgt_properties = tgt_properties.sort_values(
            by="depth", ascending=False
        )
        tgt_values = sorted_tgt_properties[by].unique() # Still sorted

    all_struct_spikes = []
    for tgt_value in tqdm(
        tgt_values,
        desc=f"Loading spikes for structure(s) `{tgt_properties.acronym.unique()}`"
    ):

        label = f"{by}: {tgt_value}"

        if by != "cluster_id":
            ids = tgt_properties[tgt_properties[by] == tgt_value].cluster_id.values
            label = f"{label}, ids={ids}"

        all_struct_spikes.append({
            'time': sorting.get_trains(
                by=by,
                tgt_values=[tgt_value]
            )[tgt_value],
            'name': label,
        })

    source = ephyviewer.InMemorySpikeSource(all_spikes=all_struct_spikes)
    view = ephyviewer.SpikeTrainViewer(source=source, name=view_name)

    window.add_view(view, location="bottom", orientation="vertical")

    return window