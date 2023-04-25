import numpy as np
import pandas as pd

from ecephys.units import SpikeInterfaceKilosortSorting, ClusterTrains


class MultiprobeSorting:
    def __init__(self, sortings: dict[str, SpikeInterfaceKilosortSorting]):
        self._sortings = sortings

    @property
    def sortings(self) -> dict[str, SpikeInterfaceKilosortSorting]:
        return self._sortings

    @property
    def probes(self) -> list[str]:
        return sorted(list(self.sortings.keys()))

    def si_cluster_ids_to_multiprobe_cluster_ids(
        self, probe: str, si_cluster_ids: np.ndarray
    ) -> np.ndarray:
        """Map probe N cluster I to a new ID: N * 100000 + I.
        Of course, this makes the implicit assumption that we will neve have more than 100k units per probe.

        It makes sense to maintain integer IDs for many comptuational tasks,
        and for compatibilty with SpikeInterface,
        so unfortuantely using ids like `imec1_id310` won't work.
        """
        prb_i = np.where(probe == np.asarray(self.probes))[0][0]
        prb_cluster_id_offset = (
            prb_i * 100000
        )  # It is unlikely we will have more than 100k units to a probe...
        return si_cluster_ids + prb_cluster_id_offset

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

    def get_trains(self) -> ClusterTrains:
        """Return ClusterTrains, keyed by multiprobe cluster IDs."""
        trains = {}
        for prb, s in self.sortings.items():
            prb_trains = s.get_trains()
            old_ids = np.asarray(list(prb_trains.keys()))
            new_ids = self.si_cluster_ids_to_multiprobe_cluster_ids(prb, old_ids)
            for old_id, new_id in zip(old_ids, new_ids):
                prb_trains[new_id] = prb_trains.pop(old_id)
            trains.update(prb_trains)
        return trains
