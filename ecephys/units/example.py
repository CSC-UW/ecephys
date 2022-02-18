import ecephys.units
from pathlib import Path


min_fr, max_fr = (0, float("inf"))
min_depth, max_depth = (0, float("inf"))

ks_dir = Path("path/to/ks/dir")

sorting = ecephys.units.load_sorting_extractor(
    ks_dir,
    good_only=False,
    drop_noise=True,
    selection_intervals={
        "fr": (min_fr, max_fr),
        "depth": (min_depth, max_depth),
    },  # Subselect clusters of interest
)  # Load spikeinterface.sortingextractor object
cluster_info = ecephys.units.get_cluster_info(ks_dir)  # All info for all clusters
cluster_info = cluster_info[
    cluster_info.cluster_id.isin(sorting.get_unit_ids())
]  # All info for subselected clusters

# Get all the spike times for each cluster
all_spike_times = ecephys.units.get_spike_times_list(sorting, sorting.get_unit_ids())
