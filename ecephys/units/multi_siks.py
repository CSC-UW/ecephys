from typing import Optional, Callable

import numpy as np
import numpy.typing as npt
import pandas as pd
from tqdm import tqdm

from ecephys.units import cluster_trains
from ecephys.units import dtypes
from ecephys.units import SpikeInterfaceKilosortSorting
import ecephys.utils


class MultiSIKS:
    """Handles SpikeInterfaceKilosortSortings from multiple probes."""

    def __init__(self, sortings: dict[str, SpikeInterfaceKilosortSorting]):
        self._sortings = sortings

    @property
    def sortings(self) -> dict[str, SpikeInterfaceKilosortSorting]:
        return self._sortings

    @property
    def probes(self) -> list[str]:
        return sorted(list(self.sortings.keys()))

    def get_probe_index(self, probe: str) -> int:
        return self.probes.index(probe)

    def si_cluster_ids_to_multiprobe_cluster_ids(
        self, probe: str, si_cluster_ids: dtypes.ClusterIDs
    ) -> dtypes.ClusterIDs:
        """Map probe N cluster I to a new ID: N * 100000 + I.
        Of course, this makes the implicit assumption that we will neve have more than 100k units per probe.

        It makes sense to maintain integer IDs for many comptuational tasks,
        and for compatibilty with SpikeInterface,
        so unfortuantely using ids like `imec1_id310` won't work.
        """
        prb_i = np.where(probe == np.asarray(self.probes))[0][0]
        prb_cluster_id_offset = int(
            prb_i * 1e6
        )  # It is unlikely we will have more than 100k units to a probe...
        return si_cluster_ids + prb_cluster_id_offset

    def si_cluster_ixs_to_multiprobe_cluster_ixs(
        self, probe: str, si_cluster_ixs: dtypes.ClusterIXs
    ) -> dtypes.ClusterIXs:
        probe_index = self.get_probe_index(probe)
        units_per_probe = [
            self.sortings[prb].si_obj.get_num_units() for prb in self.probes
        ]
        ix_offsets_per_probe = np.cumsum(np.append(0, units_per_probe))
        ix_offset = int(ix_offsets_per_probe[probe_index])
        return si_cluster_ixs + ix_offset

    @property
    def properties(self) -> pd.DataFrame:
        """Return a SpikeInterfaceKilosortSorting properties dataframe, witht he following changes:
        (1) Original, single-probe, spikeinterface cluster IDs are stored under `si_cluster_id`.
        (2) New, multiprobe cluster IDs are store under `cluster_id`.
        (3) A new `probe` columns contains the probe name, e.g. `imec1`.
        """
        props = pd.DataFrame()
        for prb, s in self.sortings.items():
            prb_props = s.properties.assign(probe=prb).rename(
                columns={"cluster_id": "si_cluster_id"}
            )
            prb_props["cluster_id"] = self.si_cluster_ids_to_multiprobe_cluster_ids(
                prb, prb_props["si_cluster_id"].to_numpy()
            )
            props = pd.concat([props, prb_props])
        return props.reset_index(drop=True)

    def get_unit_ids(self) -> dtypes.ClusterIDs:
        return self.properties["cluster_id"].values

    def multiprobe_cluster_ids_to_si_cluster_ids(
        self, multiprobe_cluster_ids: dtypes.ClusterIDs
    ) -> dtypes.ClusterIDs:
        si_cluster_ids = (
            self.properties.set_index("cluster_id")
            .loc[multiprobe_cluster_ids, "si_cluster_id"]
            .values
        )
        cluster_probes = (
            self.properties.set_index("cluster_id")
            .loc[multiprobe_cluster_ids, "probe"]
            .values
        )
        return si_cluster_ids, cluster_probes

    def refine_clusters(
        self,
        simple_filters_by_probe: Optional[dict[str, dict]] = None,
        callable_filters_by_probe: Optional[dict[str, list[Callable]]] = None,
        include_nans: bool = True,
    ):
        if simple_filters_by_probe is None:
            simple_filters_by_probe = {prb: None for prb in self.probes}
        if callable_filters_by_probe is None:
            callable_filters_by_probe = {prb: None for prb in self.probes}
        return self.__class__(
            {
                prb: sorting.refine_clusters(
                    simple_filters_by_probe[prb],
                    callable_filters_by_probe[prb],
                    include_nans,
                )
                for prb, sorting in self._sortings.items()
            }
        )

    def get_cluster_trains(
        self, display_progress: bool = True, **kwargs
    ) -> dtypes.ClusterTrains_Secs:
        """Return ClusterTrains, keyed by multiprobe cluster IDs."""
        if not kwargs.pop("return_times", True):
            raise ValueError("return_times must be True")
        mp_cluster_ids = kwargs.pop("cluster_ids", self.get_unit_ids())
        si_cluster_ids, cluster_probes = self.multiprobe_cluster_ids_to_si_cluster_ids(
            mp_cluster_ids
        )
        trains = {}
        to_load = list(zip(mp_cluster_ids, si_cluster_ids, cluster_probes))
        if display_progress:
            to_load = tqdm(to_load, desc="Loading trains by cluster_id: ")
        for mp_id, si_id, prb in to_load:
            trains[mp_id] = self.sortings[prb].get_unit_spike_train(
                si_id, return_times=True, **kwargs
            )
        return trains

    def get_spike_vector(
        self,
    ) -> tuple[dtypes.SpikeTrain_Secs, dtypes.ClusterIXs, dtypes.ClusterIDs]:
        trains = self.get_cluster_trains()
        return cluster_trains.convert_cluster_trains_to_spike_vector(trains)

    def get_trains_by_property(
        self,
        property_name: str = "acronym",
        values: Optional[
            npt.ArrayLike
        ] = None,  # Filter trains, keeping only those with the indicated property values.
        display_progress=True,
        **kwargs,
    ) -> dtypes.SpikeTrainDict:
        """Get all spike trains, merged and keyed by the desired property (e.g. depth, or structure)."""

        if not kwargs.pop("return_times", True):
            raise ValueError("return_times must be True")

        if property_name == "cluster_id":
            return self.get_cluster_trains(
                cluster_ids=values, display_progress=display_progress, **kwargs
            )

        if values is None:
            values = self.properties[property_name].unique()

        if display_progress:
            values = tqdm(values, desc=f"Loading trains by `{property_name}`: ")

        return {
            val: ecephys.utils.kway_mergesort(
                [
                    s.get_property_spike_train(
                        property_name, val, return_times=True, **kwargs
                    )
                    for s in self.sortings.values()
                ]
            )
            for val in values
        }
