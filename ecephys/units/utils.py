from pathlib import Path

import numpy as np
import pandas as pd
import spikeextractors as se


def subset_spike_times_list(
    spike_times_list, bouts_df,
):
    assert 'start_time' in bouts_df.columns
    assert 'end_time' in bouts_df.columns
    # TODO Validate that ther's no overlapping bout

    return [
        subset_spike_times(spike_times, bouts_df)
        for spike_times in spike_times_list
    ]


def subset_spike_times(spike_times, bouts_df):
    res = []
    current_concatenated_start = 0
    for i, row in bouts_df.iterrows():
        start, end = row.start_time, row.end_time
        duration = end - start
        res += [
            s - start + current_concatenated_start
            for s in spike_times
            if s >= start and s <= end
        ]
        current_concatenated_start += duration
    return res

