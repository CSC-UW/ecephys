"""For converting from SpikeInterface (si) to Pandas (pd)"""
import pandas as pd


class Units(object):
    def __init__(self, df):
        self.df = df

    def add_structure_from_sharptrack(self, sharptrack):
        """Add sharptrack structure info to self.units dataframe."""
        for structure in sharptrack.structures.itertuples():
            matches = (self.df["depth"] >= structure.lowerBorder_imec) & (
                self.df["depth"] <= structure.upperBorder_imec
            )
            self.df.loc[matches, "structure"] = structure.acronym
        self.df["structure"] = self.df["structure"].fillna("out")

    def copy(self):
        return Units(self.df.copy())

    @classmethod
    def from_spikeinterface_info(self, si_info):
        return Units(si_info.set_index("cluster_id"))


class Spikes(object):
    def __init__(self, spike_times, cluster_ids):
        """Get all spike times, labeled with cluster id, as a dataframe."""
        spikes = pd.DataFrame(
            {
                "t": spike_times,
                "cluster_id": cluster_ids,
            }
        )
        spikes.index.name = "spike"
        self.df = spikes

    # Will be unnecessary if using xarray
    def select_time(self, start_time, end_time):
        mask = (self.df["t"] >= start_time) & (self.df["t"] <= end_time)
        return Spikes(self.df.loc[mask, "t"], self.df.loc[mask, "cluster_id"])

    @classmethod
    def from_spikeinterface_sorting(self, si_sorting):
        [(spike_samples, cluster_ids)] = si_sorting.get_all_spike_trains()
        spike_times = spike_samples / si_sorting.get_sampling_frequency()
        return Spikes(spike_times, cluster_ids)
