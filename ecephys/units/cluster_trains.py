import numpy as np
from ecephys.units import dtypes


def convert_cluster_trains_to_spike_vector(
    trains: dtypes.ClusterTrains_Secs,
) -> tuple[dtypes.SpikeTrain_Secs, dtypes.ClusterIXs, dtypes.ClusterIDs]:
    """Similar to SpikeInterface's BaseSorter.to_spike_vector function.
    The spike vector concatenates all spikes of all clusters together, with the result
    being temporally sorted. This format is useful for computing cluster CCGs."""

    # Allocate output arrays
    n = np.sum([cluster_spikes.size for cluster_spikes in trains.values()])
    spike_times = np.zeros(n, dtype="float64")  # Holds time of each spike
    spike_cluster_ix = np.zeros(n, dtype="int64")  # Holds cluster index of each spike

    # Fill output arrays with unsorted data
    pos = 0
    for cluster_ix, (cluster_id, cluster_spikes) in enumerate(trains.items()):
        n = cluster_spikes.size
        spike_times[pos : pos + n] = cluster_spikes
        spike_cluster_ix[pos : pos + n] = cluster_ix
        pos += n

    # Sort
    # Setting kind='mergesort' will use Timsort under the hood.
    # Since the individual cluster spikes are already in sorted order, this is ~20-25% quicker.
    # Note: This performance benefit might not apply if we were sorting sample indices. Radiix sort would win.
    # For this reason, it may still be most efficient to get the spike vector directly from SpikeInterface, if you don't already have / need cluster trains.
    order = np.argsort(spike_times, kind="mergesort")
    spike_times = spike_times[order]
    spike_cluster_ix = spike_cluster_ix[order]

    cluster_ids = np.asarray(list(trains.keys()))
    # To get cluster_ids of each spike, simply cluster_ids[spike_cluster_ix]
    return spike_times, spike_cluster_ix, cluster_ids
