import logging

import ecephys.utils

logger = logging.getLogger(__name__)


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
