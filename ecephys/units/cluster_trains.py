import numpy as np
import numpy.typing as npt
from ecephys.units.dtypes import ClusterIDs, ClusterTrains_Secs, SpikeTrain_Secs


def convert_cluster_trains_to_spike_vector(
    trains: ClusterTrains_Secs,
) -> tuple[SpikeTrain_Secs, ClusterIDs, npt.NDArray[npt.int64]]:
    """Similar to SpikeInterface's BaseSorter.to_spike_vector function.
    The spike vector concatenates all spikes of all clusters together, with the result
    being temporally sorted. This format is useful for computing cluster CCGs."""

    # Allocate output arrays
    n = np.sum([cluster_spikes.size for cluster_spikes in trains.values()])
    spikes = np.zeros(n, dtype="float64")
    cluster_ids = np.zeros(n, dtype="int64")

    # Fill output arrays with unsorted data
    pos = 0
    for cluster_id, cluster_spikes in trains.items():
        n = cluster_spikes.size
        spikes[pos : pos + n] = cluster_spikes
        cluster_ids[pos : pos + n] = cluster_id
        pos += n

    # Sort
    # Setting kind='mergesort' will use Timsort under the hood.
    # Since the individual cluster spikes are already in sorted order, this is ~20-25% quicker.
    # Note: This performance benefit might not apply if we were sorting sample indices. Radiix sort would win.
    # For this reason, it may still be most efficient to get the spike vector directly from SpikeInterface, if you don't already have / need cluster trains.
    order = np.argsort(spikes, kind="mergesort")
    spikes = spikes[order]
    cluster_ids = cluster_ids[order]

    return spikes, cluster_ids, order
