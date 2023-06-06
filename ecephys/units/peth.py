from typing import Optional

from brainbox import singlecell
import numpy as np
import pandas as pd
import xarray as xr

from ecephys.units import ClusterTrains_Secs


def _add_cluster_properties_to_peths(
    da: xr.DataArray, props: pd.DataFrame, da_cluster_dim: str = "cluster_id"
) -> xr.DataArray:
    """Take a datarray where one dimension consists of cluster IDs, and assign to that dimension coordinates representing each cluster's properties.

    Parameters:
    -----------
    da_cluster_dim: str
        The name of the cluster ID dimension. Usually this would be `cluster_id`, but in the case of cross-correlograms it might be `clusterA` or `clusterB`.
    """
    props = props.set_index("cluster_id").loc[
        da[da_cluster_dim].values
    ]  # Order cluster_ids (i.e. rows) of properties dataframe to match datarray order
    coords = {col: (da_cluster_dim, props[col].values) for col in props}
    return da.assign_coords(coords)


def get_cluster_peths(
    trains: ClusterTrains_Secs,
    event_times: np.ndarray,
    event_labels: Optional[np.ndarray] = None,
    pre_time: float = 0.4,
    post_time: float = 0.8,
    bin_size: float = 0.025,
    return_fr: bool = True,
    cluster_properties: Optional[pd.DataFrame] = None,
) -> xr.DataArray:
    cluster_ids = list(trains.keys())
    n_bins_pre = int(np.ceil(pre_time / bin_size))
    n_bins_post = int(np.ceil(post_time / bin_size))
    n_bins = n_bins_pre + n_bins_post
    binned_spikes = np.zeros(shape=(len(cluster_ids), len(event_times), n_bins))

    for i, id in enumerate(cluster_ids):
        binned_spikes[i, :, :], tscale = singlecell.bin_spikes(
            trains[id], event_times, pre_time, post_time, bin_size
        )

    if return_fr:
        binned_spikes /= bin_size

    peths = xr.DataArray(
        binned_spikes,
        dims=("cluster_id", "event", "time"),
        coords={
            "cluster_id": cluster_ids,
            "event": event_times,
            "time": tscale,
        },
    )

    if not (event_labels is None):
        peths = peths.assign_coords({"event_type": ("event", event_labels)})

    if not (cluster_properties is None):
        peths = _add_cluster_properties_to_peths(peths, cluster_properties)

    return peths
