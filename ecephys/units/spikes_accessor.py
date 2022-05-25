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

    def as_trains(
        self, start_time=-np.Inf, end_time=np.Inf, group_spikes_by="cluster_id"
    ):
        return pd.DataFrame(
            self.select_time(start_time, end_time)
            .groupby(group_spikes_by)["t"]
            .unique()
        )

    def add_cluster_info(self, units, cols_to_add=None):
        """Return spikes df with added info from units."""
        if cols_to_add is None:
            cols_to_add = units.columns  # Add all cols from units df.

        if not all([col in units.columns] for col in cols_to_add):
            raise ValueError("Could not find all requested columns in units df.")

        units = units[cols_to_add]
        return pd.merge(
            units.reset_index(),
            self.reset_index(),
            validate="one_to_many",
        ).set_index(
            "spike"
        )  # TODO: Any way to speed this up? (Probably by specifying join key)


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
