import pandas as pd
import numpy as np
import xarray as xr
import brainbox.singlecell as bbsc
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

    @property
    def structures_by_depth(self):
        return self.units.sort_values("depth").structure.unique().tolist()

    def copy(self):
        return SingleProbeSorting(
            self.spikes.copy(),
            self.units.copy(),
        )

    def subset_units(self, cluster_ids_to_keep):
        self.spikes = self.spikes.loc[
            self.spikes["cluster_id"].isin(sorted(cluster_ids_to_keep))
        ]
        self.units = self.units.loc[sorted(cluster_ids_to_keep)]

    def filter_trains_by_structure(self, desired_structures):
        """Given a list of desired structures, return only trains belonging to those structures."""
        trains = self.spikes.spikes.as_trains()
        cluster_structures = self.units["structure"]
        labeled_trains = trains.join(cluster_structures)
        is_desired = np.isin(labeled_trains["structure"], desired_structures)
        return labeled_trains[is_desired]

    # TODO: There's probably ways to make this faster
    # TODO: Aggregate other columns when units-to-train merge is not many-to-1
    def get_spike_trains_for_plotting(
        self, start_time=-float("Inf"), end_time=float("Inf"), grouping_col="cluster_id"
    ):
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
        if grouping_col == "depth":
            DF_DEPTH_STEP = 20
            observed_depths = spikes.depth.values
            tmp = np.diff(np.sort(spikes.depth.values))
            observed_depth_step = tmp[tmp > 0].min()
            depth_step = min(DF_DEPTH_STEP, observed_depth_step)
            depth_categories = np.arange(
                min(observed_depths), max(observed_depths) + depth_step, depth_step
            )
            spikes["depth"] = spikes["depth"].astype("category")
            spikes["depth"] = spikes["depth"].cat.set_categories(
                depth_categories, ordered=True
            )

        trains = spikes.spikes.as_trains(
            start_time=start_time,
            end_time=end_time,
            grouping_col=grouping_col,
        )

        # Add all other columns from units after turning into trains
        if grouping_col == "cluster_id":
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

    def get_peths(
        self, events, include_cluster_structure=True, include_cluster_group=True
    ):
        # events: DataFrame, must have column 'ephys_time' with the time of each event.

        # This takes ~90 seconds per 1000 trials for a single probe
        # This would be trivial to turn into a general get_peths() method
        cluster_ids = self.spikes["cluster_id"].unique()
        try:
            _peths, binned_spikes = bbsc.calculate_peths(
                spike_times=self.spikes["t"],
                spike_clusters=self.spikes["cluster_id"],
                cluster_ids=cluster_ids,
                align_times=events["ephys_time"],
                pre_time=1.0,
                post_time=1.0,
                bin_size=0.025,
                smoothing=0,
                return_fr=True,
            )
        except ValueError:
            pass

        # Wrap returned data in DataArrays for convenience.
        peths = xr.DataArray(
            data=binned_spikes,
            dims=["event", "cluster_id", "bin_time"],
            coords={
                "event": events.index.values,
                "cluster_id": _peths.cscale,
                "bin_time": _peths.tscale,
            },
        )  # bin centers

        # Add other event properties, like ephys time, event type, or sleep/wake state
        for event_property in events.columns:
            peths = peths.assign_coords(
                {event_property: ("event", events[event_property])}
            )

        # Add certain unit properties, if requested
        if include_cluster_structure:
            peths = peths.assign_coords(
                {
                    "structure": (
                        "cluster_id",
                        self.units.loc[cluster_ids, "structure"],
                    ),
                }
            )

        if include_cluster_group:
            peths = peths.assign_coords(
                {
                    "group": ("cluster_id", self.units.loc[cluster_ids, "group"]),
                }
            )

        return peths


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

    @property
    def structures_by_depth(self):
        return {
            probe: sorting.structures_by_depth for probe, sorting in self.sorts.items()
        }

    def copy(self):
        return MultiProbeSorting(
            {prb: sorting.copy() for prb, sorting in self.sorts.items()}
        )

    def subset_units(self, cluster_ids_to_keep_per_probe):
        for probe, cluster_ids_to_keep in cluster_ids_to_keep_per_probe.items():
            self.sorts[probe].subset_units(cluster_ids_to_keep)

    def filter_trains_by_structure(self, desired_structures):
        """Takes a dict where each key is a probe (e.g. 'imec0'), and each value is the list of structures you want trains from."""
        return pd.concat(
            [
                self.sorts[probe].filter_trains_by_structure(structures)
                for probe, structures in desired_structures.items()
            ],
            keys=desired_structures.keys(),
            names=["probe"],
        ).reset_index()

    def get_spike_trains_for_plotting(
        self, start_time=-float("Inf"), end_time=float("Inf"), grouping_col="cluster_id"
    ):
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
