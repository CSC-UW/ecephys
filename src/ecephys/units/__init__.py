# Only import lightweight submodules at package level.
# Heavy modules (siks_sorting, correlograms, multi_siks, siutils) must be
# imported directly to avoid slow import times from matplotlib, spikeinterface, sklearn:
#   from ecephys.units import siks_sorting
#   from ecephys.units.siks_sorting import SpikeInterfaceKilosortSorting
from . import binning, cluster_trains, dtypes, peths

from .binning import bin_train, bin_trains, get_aligned_bins
from .cluster_trains import convert_cluster_trains_to_spike_vector
from .peths import (
    get_peths_from_trains,
    get_peths_sem,
    zscore_peths_by_peri_event_window,
)

__all__ = [
    "binning",
    "cluster_trains",
    "dtypes",
    "peths",
    "bin_train",
    "bin_trains",
    "get_aligned_bins",
    "convert_cluster_trains_to_spike_vector",
    "get_peths_from_trains",
    "get_peths_sem",
    "zscore_peths_by_peri_event_window",
]
