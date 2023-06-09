import numpy as np
import pandas as pd
from tqdm import tqdm

from ecephys.units import cluster_trains
from ecephys.units import dtypes
from ecephys.units import SpikeInterfaceKilosortSorting


class MultiprobeSorting:
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

    def refine_clusters(
        self, filters_by_probe: dict[str, dict], include_nans: bool = True
    ):
        return self.__class__(
            {
                prb: sorting.refine_clusters(
                    filters_by_probe[prb], include_nans=include_nans
                )
                for prb, sorting in self._sortings.items()
            }
        )

    def get_cluster_trains(self, **kwargs) -> dtypes.ClusterTrains_Secs:
        """Return ClusterTrains, keyed by multiprobe cluster IDs."""
        trains = {}
        for prb in self.probes:
            prb_trains = self.sortings[prb].get_cluster_trains(
                return_times=True, **kwargs
            )
            old_ids = np.asarray(list(prb_trains.keys()))
            new_ids = self.si_cluster_ids_to_multiprobe_cluster_ids(prb, old_ids)
            for old_id, new_id in zip(old_ids, new_ids):
                prb_trains[new_id] = prb_trains.pop(old_id)
            trains.update(prb_trains)
        return trains

    def get_spike_vector(
        self,
    ) -> tuple[dtypes.SpikeTrain_Secs, dtypes.ClusterIXs, dtypes.ClusterIDs]:
        trains = self.get_cluster_trains()
        return cluster_trains.convert_cluster_trains_to_spike_vector(trains)
