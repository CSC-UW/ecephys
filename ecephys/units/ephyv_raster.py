import ephyviewer
from tqdm import tqdm
import numpy as np
from ecephys.plot import state_colors
import matplotlib.colors


DEPTH_STEP = 20


def plot_interactive_ephyviewer_raster(si_ks_sorting, by="cluster_id"):

    app = ephyviewer.mkQApp()

    # #Create the main window that can contain several viewers
    win = ephyviewer.MainViewer(debug=True, show_auto_scale=True)

    # Add hypnogram ayyy
    if si_ks_sorting.hypnogram is not None:

        hyp = si_ks_sorting.hypnogram
        all_states = hyp.state.unique()
        all_epochs = []

        for state in all_states:
            mask = hyp["state"] == state
            all_epochs.append({
                "name": state,
                "label": np.array([state for _ in range(mask.sum())]),
                "time": hyp[mask].start_time.values,
                "duration": hyp[mask].duration.values,
            })

        source_epochs = ephyviewer.InMemoryEpochSource(all_epochs=all_epochs)
        view = ephyviewer.EpochViewer(source=source_epochs, name='Hypnogram')

        # Set usual Hypnogram colors
        for i, state in enumerate(all_states):
            view.by_channel_params[f"ch{i}", 'color'] = matplotlib.colors.rgb2hex(state_colors[state])

        win.add_view(view, location="bottom", orientation="vertical")

        # Add event list for navigation
        view = ephyviewer.EventList(source=source_epochs, name='event')
        win.add_view(view, location='bottom',  orientation='horizontal')

    # Iterate on structures by mean depth
    properties = si_ks_sorting.properties
    group_means = properties.groupby("acronym")["depth"].mean().reset_index()
    sorted_means = group_means.sort_values(by="depth", ascending=False)

    for _, struct_row in sorted_means.iterrows():
        struct = struct_row["acronym"]
        mask = properties["acronym"] == struct
        struct_properties = properties[mask]

        if by == "depth":
            tgt_values = np.arange(
                struct_properties.depth.min(),
                struct_properties.depth.max() + DEPTH_STEP,
                DEPTH_STEP
            )[::-1] # Descending depths between structure min/max in 20um steps
        else:
            # Sort units by depth within each structure
            sorted_struct_properties = struct_properties.sort_values(
                by="depth", ascending=False
            )
            tgt_values = sorted_struct_properties[by].unique() # Still sorted

        all_struct_spikes = []
        for tgt_value in tqdm(
            tgt_values,
            desc=f"Loading spikes for structure `{struct}`"
        ):

            label = f"{by}: {tgt_value}"
            
            if by != "cluster_id":
                ids = struct_properties[struct_properties[by] == tgt_value].cluster_id.values
                label = f"{label}, ids={ids}"

            train = si_ks_sorting.get_trains(
                by=by,
                tgt_values=[tgt_value]
            )[tgt_value]
            all_struct_spikes.append({
                'time': train,
                'name': label,
            })

        source = ephyviewer.InMemorySpikeSource(all_spikes=all_struct_spikes)
        view = ephyviewer.SpikeTrainViewer(source=source, name=f"Structure: {struct}")

        win.add_view(view, location="bottom", orientation="vertical")

    win.show()
    app.exec()