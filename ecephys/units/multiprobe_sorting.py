import numpy as np
import pandas as pd

from ecephys import utils
from ecephys.units import dtypes
from ecephys.units import SpikeInterfaceKilosortSorting


class MultiprobeSorting:
    def __init__(self, sortings: dict[str, SpikeInterfaceKilosortSorting]):
        self._sortings = sortings

        try:
            self.get_num_segments()
        except AssertionError:
            raise ValueError("All sortings must have the same number of segments")

    @property
    def sortings(self) -> dict[str, SpikeInterfaceKilosortSorting]:
        return self._sortings

    @property
    def probes(self) -> list[str]:
        return sorted(list(self.sortings.keys()))

    def get_num_segments(self):
        segments_per_sorting = [
            s.si_obj.get_num_segments() for prb, s in self.sortings.items()
        ]
        assert utils.all_equal(
            segments_per_sorting
        ), "All sortings must have the same number of segments"
        return segments_per_sorting[0]

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
        for prb, s in self.sortings.items():
            prb_trains = s.get_cluster_trains(**kwargs)
            old_ids = np.asarray(list(prb_trains.keys()))
            new_ids = self.si_cluster_ids_to_multiprobe_cluster_ids(prb, old_ids)
            for old_id, new_id in zip(old_ids, new_ids):
                prb_trains[new_id] = prb_trains.pop(old_id)
            trains.update(prb_trains)
        return trains

    def get_all_spike_trains(self, outputs="unit_id"):
        spikes_by_probe = []
        for prb, s in self.sortings.items():
            prb_spikes = s.get_all_spike_trains(outputs, return_times=True)
            for segment_index in range(len(prb_spikes)):
                if outputs == "unit_id":
                    prb_spikes[segment_index][
                        1
                    ] = self.si_cluster_ids_to_multiprobe_cluster_ids(
                        prb, prb_spikes[segment_index][1]
                    )
                elif outputs == "unit_index":
                    prb_spikes[
                        segment_index[1]
                    ] = self.si_cluster_ixs_to_multiprobe_cluster_ixs(
                        prb, prb_spikes[segment_index][1]
                    )
                else:
                    raise ValueError(f"Unrecognized output format: {outputs}")
            spikes_by_probe.append(prb_spikes)

        spikes = []
        for segment_index in range(self.get_num_segments()):
            raise NotImplementedError("How to handle merging of spike labels?")
            spikes[segment_index][0] = utils.kway_sortednp_merge(
                [prb_spikes[segment_index][0] for prb_spikes in spikes_by_probe]
            )

        return spikes
