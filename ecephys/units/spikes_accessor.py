import pandas as pd
import numpy as np


@pd.api.extensions.register_dataframe_accessor("spikes")
class SpikesAccessor:
    def __init__(self, df):
        self._df = df
        self.validate()

    def validate(self):
        if "t" not in self._df.columns:
            raise ValueError("`t` column not found.")
        if "cluster_id" not in self._df.columns:
            raise ValueError("`cluster_id` column not found.")
        if "spike" != self._df.index.name:
            raise ValueError("`spike` index not found.")

    def select_time(self, start_time, end_time):
        mask = (self._df["t"] >= start_time) & (self._df["t"] <= end_time)
        return self._df.loc[mask]

    def as_trains(self, start_time=-np.Inf, end_time=np.Inf, grouping_col="cluster_id"):
        if not grouping_col in self._df.columns:
            raise ValueError(f"Can't find grouping col `{grouping_col}` in spikes df to generate trains.")
        return (
            self.select_time(start_time, end_time).groupby(grouping_col)["t"].unique()
        )


def spikeinterface_sorting_to_dataframe(si_sorting):
    [(spike_samples, cluster_ids)] = si_sorting.get_all_spike_trains()
    spike_times = spike_samples / si_sorting.get_sampling_frequency()
    df = pd.DataFrame(
        {
            "t": spike_times,
            "cluster_id": cluster_ids,
        }
    )
    df.index.name = "spike"
    return df
