import pandas as pd
import numpy as np
import xarray as xr
import brainbox.singlecell as bbsc
from abc import ABC, abstractproperty, abstractmethod
from .plot import set_uniform_rgba


class Sorting(ABC):
    @abstractproperty
    def firstSpikeTime(self):
        pass

    @abstractproperty
    def lastSpikeTime(self):
        pass

    @abstractproperty
    def nClusters(self):
        pass

    @abstractmethod
    def get_spike_trains_for_plotting(self, start_time, end_time):
        pass


class SingleProbeSorting(Sorting):
    def __init__(self, siSorting, timeConverter=None):
        """
        Parameters:
        ===========
        siSorting: UnitsSelectionSorting
        timeConverter: function
            Takes an array of spike times t, and returns a new array of corresponding spike times.
            This is used to map spike times from one probe to another, since the SpikeInterface objects can not do this.
        """
        self._si = siSorting
        self._timeConverter = timeConverter
        self.spikes = self.spikeinterface_sorting_to_dataframe(siSorting, timeConverter)

    def __repr__(self):
        return repr(self.si)

    @property
    def si(self):
        return self._si

    @property
    def firstSpikeTime(self):
        return self.spikes["t"].min()

    @property
    def lastSpikeTime(self):
        return self.spikes["t"].max()

    @property
    def clusterInfo(self):
        clusterInfo = pd.DataFrame(self.si._properties)
        clusterInfo["cluster_id"] = self.si.get_unit_ids()
        return clusterInfo

    @property
    def nClusters(self):
        return len(self.si.get_unit_ids())

    @property
    def structuresByDepth(self):
        return self.clusterInfo.sort_values("depth")["structure"].unique().tolist()

    def clone(self):
        return SingleProbeSorting(self.si.clone(), self._timeConverter)

    def get_spike_trains_for_plotting(self, start_time, end_time):
        pass

    @staticmethod
    def spikeinterface_sorting_to_dataframe(siSorting, timeConverter=None):
        [(spikeSamples, clusterIDs)] = siSorting.get_all_spike_trains()
        spikeTimes = spikeSamples / siSorting.get_sampling_frequency()
        spikes = pd.DataFrame(
            {
                "t": spikeTimes,
                "cluster_id": clusterIDs,
            }
        )
        if timeConverter is not None:
            spikes["t"] = timeConverter(spikes["t"].values)
        return spikes


class MultiProbeSorting(Sorting):
    def __init__(self, single_probe_sortings):
        self.sortings = single_probe_sortings

    @property
    def firstSpikeTime(self):
        return min(sorting.firstSpikeTime for probe, sorting in self.sortings.items())

    @property
    def lastSpikeTime(self):
        return max(sorting.lastSpikeTime for probe, sorting in self.sortings.items())

    @property
    def nClusters(self):
        return sum(sorting.nClusters for probe, sorting in self.sortings.items())

    @property
    def structuresByDepth(self):
        return {
            probe: sorting.structuresByDepth for probe, sorting in self.sortings.items()
        }

    def clone(self):
        return MultiProbeSorting(
            {probe: sorting.clone() for probe, sorting in self.sortings.items()}
        )

    def get_spike_trains_for_plotting(self, start_time, end_time):
        pass
