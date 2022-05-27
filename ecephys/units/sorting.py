import pandas as pd
import numpy as np
from abc import ABC, abstractproperty, abstractmethod
from matplotlib.colors import to_rgba
from .plot import set_uniform_rgba


class Sorting(ABC):
    @abstractproperty
    def data_start(self):
        pass

    @abstractproperty
    def data_end(self):
        pass

    @abstractproperty
    def n_units(self):
        pass

    @abstractmethod
    def get_spike_trains_for_plotting(self, start_time, end_time):
        pass


class SingleProbeSorting(Sorting):
    def __init__(self, spikes, units):
        self.spikes = spikes
        self.units = units

    @property
    def data_start(self):
        return self.spikes.t.min()

    @property
    def data_end(self):
        return self.spikes.t.max()

    @property
    def n_units(self):
        return len(self.units)

    def get_spike_trains_for_plotting(self, start_time=None, end_time=None):
        start_time = self.data_start if start_time is None else start_time
        end_time = self.data_end if end_time is None else end_time

        trains = self.spikes.spikes.as_trains(start_time, end_time)

        if "rgba" not in self.units.columns:
            set_uniform_rgba(self.units, "black")

        trains = self.units.join(trains, how="outer")
        silent = trains.trains.silent()
        # Add ghost spikes at very start and end of window to silent trains, to reserve space for them on the plot's x and y axes.
        trains.loc[silent, "t"] = pd.Series(
            [np.array((start_time, end_time))] * sum(silent), dtype="float64"
        ).values
        # Make silent units white and transparent, so that they are invisible.
        trains.loc[silent, "rgba"] = pd.Series(
            [to_rgba("white", 0.0)] * sum(silent), dtype="object"
        ).values
        return trains.sort_values("depth")


class MultiProbeSorting(Sorting):
    def __init__(self, single_probe_sortings):
        self.sorts = single_probe_sortings

    @property
    def data_start(self):
        return min(self.sorts[probe].spikes.t.min() for probe in self.sorts)

    @property
    def data_end(self):
        return max(self.sorts[probe].spikes.t.max() for probe in self.sorts)

    @property
    def n_units(self):
        return sum(len(self.sorts[probe].units) for probe in self.sorts)

    def get_spike_trains_for_plotting(self, start_time=None, end_time=None):
        start_time = self.data_start if start_time is None else start_time
        end_time = self.data_end if end_time is None else end_time

        trains = {
            probe: self.sorts[probe].get_spike_trains_for_plotting(
                start_time,
                end_time,
            )
            for probe in self.sorts
        }

        return pd.concat(
            [trains[probe] for probe in trains], keys=trains.keys(), names=["probe"]
        ).reset_index()
