import ecephys
import logging
import numpy as np
from pandas.api import types

logger = logging.getLogger(__name__)


def refine_clusters(si_obj, filters={}):
    """Subselect clusters based on filters.

    Parameters:
    ===========
    si_obj: SI extractor or sorting object
    filters: dict
        Keys are the names of cluster properties.  Example properties include all columns in cluster_info.tsv, including quality metrics.
        Values are either 2-item tuples or sets.
        Tuples specify a range of allowable values for non-categorical numeric properties.
        Sets specify allowable values for categorical properties.
        For example, {"n_spikes": (2, np.Inf)} will load only clusters with 2 or more spikes.
        For example, {"quality": {"good", "mua"}} will load only clusters marked as such after curation and QMs.

    Returns:
    ========
    UnitsSelectionSorting

    Notes:
    ======
    SpikeInterface renames the 'group' columns in cluster_info.tsv to 'quality'.
    Tom previously created an 'unsorted' category of 'quality', which is now 'NaN' (or whatever the default is)
    """
    keep = np.ones_like(si_obj.get_unit_ids())
    for property, filter in filters.items():
        if not property in si_obj.get_property_keys():
            logger.warn(
                f"Cluster property {property} not found. Unable to filter clusters based on {property}."
            )
            continue
        if isinstance(filter, tuple):
            lo, hi = filter
            values = si_obj.get_property(property)
            assert types.is_numeric_dtype(
                values.dtype
            ), f"Cannot select a range of values for cluster property {property} with dtype {values.dtype}. Expected a numeric dtype."
            mask = np.logical_and(values >= lo, values <= hi)
            keep = keep & mask
        elif isinstance(filter, set):
            mask = np.isin(si_obj.get_property(property), list(filter))
            keep = keep & mask
        else:
            raise ValueError(
                f"Cluster property {property} was provided as type {type(filter)}. Expected a tuple for selecting a range of numerical values, or a set for selecting categorical variables."
            )
        print(f"{property}: {filter} excludes {mask.size - mask.sum()} clusters.")

    print(
        f"{keep.size - keep.sum()}/{keep.size} clusters excluded by jointly applying filters."
    )
    clusterIDs = si_obj.get_unit_ids()[np.where(keep)]
    return si_obj.select_units(clusterIDs)


# TODO: Rename lowerBorder_imec to something non-neuropixel specific.
#   The only real requirement is that the units match those of the sorting object's depth property.
def add_structures_from_sharptrack(sorting, sharptrack):
    """Use a SHARPTrack object to add a `structure` property to a SpikeInterface sorting object.

    Parameters:
    ===========
    sorting: UnitsSelectionSorting
        Must contain a 'depth' column
    sharptrack: ecephys.SHARPTrack
    """
    depths = sorting.get_property("depth")
    structures = np.empty(depths.shape, dtype=object)
    for structure in sharptrack.structures.itertuples():
        lo = structure.lowerBorder_imec
        hi = structure.upperBorder_imec
        mask = (depths >= lo) & (depths <= hi)
        structures[np.where(mask)] = structure.acronym
    sorting.set_property("structure", structures)


# TODO: Is this used, and for what, and by whom? Does it belong in a project repo?
def get_spike_times(extr, cluster_id):
    sf = extr.get_sampling_frequency()
    return [f / sf for f in extr.get_unit_spike_train(unit_id=cluster_id)]


# TODO: Is this used, and for what, and by whom? Does it belong in a project repo?
def get_spike_times_list(extr, cluster_ids=None):
    if cluster_ids is None:
        cluster_ids = extr.get_unit_ids()
    return [get_spike_times(extr, cluster_id) for cluster_id in cluster_ids]


# TODO: Is this used, and for what, and by whom? Does it belong in a project repo?
def pool_spike_times_list(spike_times_list):
    return sorted(ecephys.utils.flatten(spike_times_list))


# TODO: Is this used, and for what, and by whom? Does it belong in a project repo?
def subset_spike_times_list(
    spike_times_list,
    bouts_df,
):
    assert "start_time" in bouts_df.columns
    assert "end_time" in bouts_df.columns
    # TODO Validate that ther's no overlapping bout

    return [
        subset_spike_times(spike_times, bouts_df) for spike_times in spike_times_list
    ]


# TODO: Is this used, and for what, and by whom? Does it belong in a project repo?
def subset_spike_times(spike_times, bouts_df):
    res = []
    current_concatenated_start = 0
    for row in bouts_df.itertuples():
        start, end = row.start_time, row.end_time
        duration = end - start
        res += [
            s - start + current_concatenated_start
            for s in spike_times
            if s >= start and s <= end
        ]
        current_concatenated_start += duration
    return res
