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

    # TODO: There's probably ways to make this faster
    # TODO: Aggregate other columns when units-to-train merge is not many-to-1
    def get_spike_trains_for_plotting(self, start_time=-float('Inf'), end_time=float('Inf'), grouping_col='cluster_id'):
        if grouping_col not in self.spikes:
            # Add groupby column from units to spikes data
            spikes = self.spikes.spikes.join_units(
                self.units,
                units_columns=[grouping_col],
                start_time=start_time,
                end_time=end_time,
            )
        else:
            spikes = self.spikes

        # Turn depth column to categorical so we represent "empty" depths when grouping as trains
        if grouping_col == 'depth':
            DF_DEPTH_STEP = 20
            observed_depths = spikes.depth.values 
            tmp = np.diff(np.sort(spikes.depth.values))
            observed_depth_step = tmp[tmp>0].min()
            depth_step = min(DF_DEPTH_STEP, observed_depth_step)
            depth_categories = np.arange(min(observed_depths), max(observed_depths) + depth_step, depth_step)
            spikes['depth'] = spikes['depth'].astype("category")
            spikes['depth'] = spikes['depth'].cat.set_categories(depth_categories, ordered=True)

        trains = spikes.spikes.as_trains(
            start_time=start_time,
            end_time=end_time,
            grouping_col=grouping_col,
        )

        # Add all other columns from units after turning into trains
        if grouping_col == 'cluster_id':
            trains = trains.trains.join_units(self.units)

        if "rgba" not in trains.columns:
            set_uniform_rgba(trains, "black")

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

    def get_spike_trains_for_plotting(self, start_time=-float('Inf'), end_time=float('Inf'), grouping_col='cluster_id'):
        start_time = self.data_start if start_time is None else start_time
        end_time = self.data_end if end_time is None else end_time

        trains = {
            probe: self.sorts[probe].get_spike_trains_for_plotting(
                start_time=start_time,
                end_time=end_time,
                grouping_col=grouping_col,
            )
            for probe in self.sorts
        }

        return pd.concat(
            [trains[probe] for probe in trains], keys=trains.keys(), names=["probe"]
        ).reset_index()
