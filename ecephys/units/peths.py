from typing import Optional

import numba
import numba.typed
import numba.types
import numpy as np
import pandas as pd
import xarray as xr

from ecephys.units import dtypes


def get_peths_from_trains(
    trains: dtypes.ClusterTrains_Secs,
    event_times: np.ndarray,
    event_labels: Optional[np.ndarray] = None,
    pre_time: float = 0.4,
    post_time: float = 0.8,
    bin_size: float = 0.025,
    return_fr: bool = True,
    property_frame: Optional[pd.DataFrame] = None,
    property_names: Optional[list[str]] = None,
) -> xr.DataArray:
    """Get PETHs, using numba to speed up the loop.
    If you already have a spike vector (costly, but it happens), use get_peths_from_spike_vector instead. It is lighting fast.
    If you have small data (fewer spikes or fewer clusters), use get_peths_from_trains_v2 instead. It has less overhead because it doens't parallelize.

    Compared to get_peths_from_trains_v2, which uses bin_single_spiketrain_numba, this...
      - parallelizes over trains
      - deduplicates the redundant work done every loop
      - uses only a single pre-allocated array
      - uses a numba.typed.dict

    For 48h, ~600 clusters:
        - IBL approach: 3m40s
        - This approach: 13.2s
        - v2 approach (no parallelization): 22.2s

    We keep v2 around because it is simpler and faster for small data, but also because the numba typed dict, which this relies on, may be unstable or lose support.
    """
    numba_trains = numba.typed.Dict.empty(numba.types.int64, numba.types.float64[:])
    for id, train in trains.items():
        numba_trains[id] = train
    binned_spikes, tscale, cluster_ids = bin_spiketrains_numba(
        numba_trains, event_times, pre_time, post_time, bin_size
    )

    if return_fr:
        binned_spikes /= bin_size

    peths = xr.DataArray(
        binned_spikes,
        dims=("cluster_id", "event", "time"),
        coords={
            "cluster_id": cluster_ids,
            "event": event_times,
            "time": tscale,
        },
    )

    if not (event_labels is None):
        peths = peths.assign_coords({"event_type": ("event", event_labels)})

    if not (property_frame is None):
        peths = add_cluster_properties_to_peths(peths, property_frame, property_names)

    return peths


@numba.njit(parallel=True, nogil=True, cache=True)
def bin_spiketrains_numba(numba_trains, event_times, pre_time, post_time, bin_size):
    n_bins_pre = int(np.ceil(pre_time / bin_size))
    n_bins_post = int(np.ceil(post_time / bin_size))
    n_bins = n_bins_pre + n_bins_post
    tscale = np.arange(-n_bins_pre, n_bins_post + 1) * bin_size

    ts = np.repeat(event_times, tscale.size).reshape(-1, tscale.size) + tscale
    epoch_bounds = np.zeros(shape=(event_times.size, 2))
    epoch_bounds[:, 0] = ts[:, 0]
    epoch_bounds[:, 1] = ts[:, -1]

    cluster_ids = np.asarray(list(numba_trains.keys()))
    binned_spikes = np.zeros(
        shape=(cluster_ids.size, event_times.size, n_bins), dtype=np.float64
    )
    for i_unit in numba.prange(cluster_ids.size):
        id = cluster_ids[i_unit]
        unit_spike_times = numba_trains[id]
        epoch_idxs = np.searchsorted(unit_spike_times, epoch_bounds)
        for i_event, (ep, t) in enumerate(zip(epoch_idxs, ts)):
            xind = (
                np.floor((unit_spike_times[ep[0] : ep[1]] - t[0]) / bin_size)
            ).astype(np.int64)
            r = np.bincount(xind, minlength=tscale.shape[0])
            binned_spikes[i_unit, i_event, :] = r[:-1]

    tscale = (tscale[:-1] + tscale[1:]) / 2

    return binned_spikes, tscale, cluster_ids


def add_cluster_properties_to_peths(
    da: xr.DataArray,
    property_frame: pd.DataFrame,
    property_names: Optional[list[str]] = None,
) -> xr.DataArray:
    """Take a datarray where one dimension consists of cluster IDs, and assign to that dimension coordinates representing each cluster's properties.

    Parameters:
    -----------
    da_cluster_dim: str
        The name of the cluster ID dimension. Usually this would be `cluster_id`, but in the case of cross-correlograms it might be `clusterA` or `clusterB`.
    """
    property_frame = property_frame.set_index("cluster_id").loc[
        da["cluster_id"].values
    ]  # Order cluster_ids (i.e. rows) of properties dataframe to match datarray order
    property_names = (
        property_frame.columns if property_names is None else property_names
    )
    coords = {col: ("cluster_id", property_frame[col].values) for col in property_names}
    return da.assign_coords(coords)


################
# Niche functions, you'll probably never use.
################


def get_peths_from_trains_alt(
    trains: dtypes.ClusterTrains_Secs,
    event_times: np.ndarray,
    event_labels: Optional[np.ndarray] = None,
    pre_time: float = 0.4,
    post_time: float = 0.8,
    bin_size: float = 0.025,
    return_fr: bool = True,
    property_frame: Optional[pd.DataFrame] = None,
    property_names: Optional[list[str]] = None,
) -> xr.DataArray:
    """Get PETHs, using numba to speed up the loop.
    If you have big data, use get_peths_from_trains instead. It parallelizes over trains.
    If you already have a spike vector (costly, but it happens), use get_peths_from_spike_vector instead. It is lighting fast.

    Keep this function around, because it uses only stable functionality, and if you can't use numba, you can drop in singlecell.bin_spikes.
    """
    cluster_ids = np.asarray(list(trains.keys()))
    n_bins_pre = int(np.ceil(pre_time / bin_size))
    n_bins_post = int(np.ceil(post_time / bin_size))
    n_bins = n_bins_pre + n_bins_post
    binned_spikes = np.zeros(
        shape=(cluster_ids.size, event_times.size, n_bins), dtype=np.float64
    )

    for i, id in enumerate(cluster_ids):
        binned_spikes[i], tscale = bin_single_spiketrain_numba(
            trains[id], event_times, pre_time, post_time, bin_size
        )
        # binned_spikes[i], tscale = singlecell.bin_spikes(trains[id], event_times, pre_time, post_time, bin_size)

    if return_fr:
        binned_spikes /= bin_size

    peths = xr.DataArray(
        binned_spikes,
        dims=("cluster_id", "event", "time"),
        coords={
            "cluster_id": cluster_ids,
            "event": event_times,
            "time": tscale,
        },
    )

    if not (event_labels is None):
        peths = peths.assign_coords({"event_type": ("event", event_labels)})

    if not (property_frame is None):
        peths = add_cluster_properties_to_peths(peths, property_frame, property_names)

    return peths


@numba.jit(
    (numba.float64[:], numba.float64[:], numba.float64, numba.float64, numba.float64),
    nopython=True,
    nogil=True,
    cache=True,
)
def bin_single_spiketrain_numba(
    spike_times, event_times, pre_time, post_time, bin_size
):
    """Based on brainbox.singlecell.bin_spikes, but using numba to speed up the loop.
    Removed weights for simplicity and speed, since we never use them anyways.

    Some of the functions used in brainbox.singlecell.bin_spikes are not supported by numba,
    so we use functional equivalents (e.g. reshape instead of axis kwargs, new array vs np.c_).
    """
    n_bins_pre = int(np.ceil(pre_time / bin_size))
    n_bins_post = int(np.ceil(post_time / bin_size))
    n_bins = n_bins_pre + n_bins_post

    tscale = np.arange(-n_bins_pre, n_bins_post + 1) * bin_size
    ts = np.repeat(event_times, tscale.size).reshape(-1, tscale.size) + tscale
    epoch_bounds = np.zeros(shape=(event_times.size, 2))
    epoch_bounds[:, 0] = ts[:, 0]
    epoch_bounds[:, 1] = ts[:, -1]

    binned_spikes = np.zeros(shape=(event_times.size, n_bins), dtype=np.float64)
    epoch_idxs = np.searchsorted(spike_times, epoch_bounds)
    for i_ep, (ep, t) in enumerate(zip(epoch_idxs, ts)):
        xind = (np.floor((spike_times[ep[0] : ep[1]] - t[0]) / bin_size)).astype(
            np.int64
        )
        r = np.bincount(xind, minlength=tscale.shape[0])
        binned_spikes[i_ep, :] = r[:-1]

    tscale = (tscale[:-1] + tscale[1:]) / 2
    return binned_spikes, tscale


def get_peths_from_spike_vector(
    spike_times: dtypes.SpikeTrain_Secs,
    spike_cluster_ixs: dtypes.ClusterIXs,
    cluster_ids: dtypes.ClusterIDs,
    event_times: np.ndarray[np.float64],
    event_labels: Optional[np.ndarray] = None,
    pre_time: float = 0.4,
    post_time: float = 0.8,
    bin_size: float = 0.025,
    return_fr: bool = True,
    property_frame: Optional[pd.DataFrame] = None,
    property_names: Optional[list[str]] = None,
) -> xr.DataArray:
    """Creating the spike vector from trains can be costly, but if you have done it, this is fast.
    For ~48h, ~600 clusters, ~50,000 events, it takes less than 50s.

    Usage:
    spike_times, spike_cluster_ixs, cluster_ids = units.convert_cluster_trains_to_spike_vector(trains)
    peths = get_peths_from_spike_vector(spike_times, spike_cluster_ixs, cluster_ids, event_times)
    """
    binned_spikes, tscale = bin_spike_vector_numba(
        spike_times,
        spike_cluster_ixs,
        cluster_ids,
        event_times,
        pre_time,
        post_time,
        bin_size,
    )

    if return_fr:
        binned_spikes /= bin_size

    peths = xr.DataArray(
        binned_spikes,
        dims=("cluster_id", "event", "time"),
        coords={
            "cluster_id": cluster_ids,
            "event": event_times,
            "time": tscale,
        },
    )

    if not (event_labels is None):
        peths = peths.assign_coords({"event_type": ("event", event_labels)})

    if not (property_frame is None):
        peths = add_cluster_properties_to_peths(peths, property_frame, property_names)

    return peths


@numba.jit(
    (
        numba.float64[:],
        numba.int64[:],
        numba.int64[:],
        numba.float64[:],
        numba.float64,
        numba.float64,
        numba.float64,
    ),
    nopython=True,
    nogil=True,
    cache=True,
    parallel=True,
)
def bin_spike_vector_numba(
    spike_times,
    spike_cluster_ixs,
    cluster_ids,
    event_times,
    pre_time,
    post_time,
    bin_size,
):
    n_bins_pre = int(np.ceil(pre_time / bin_size))
    n_bins_post = int(np.ceil(post_time / bin_size))
    n_bins = n_bins_pre + n_bins_post
    tscale = np.arange(-n_bins_pre, n_bins_post + 1) * bin_size
    ts = np.repeat(event_times, tscale.size).reshape(-1, tscale.size) + tscale
    epoch_bounds = np.zeros(shape=(event_times.shape[0], 2))
    epoch_bounds[:, 0] = ts[:, 0]
    epoch_bounds[:, 1] = ts[:, -1]
    epoch_bounds
    binned_spikes = np.zeros(shape=(cluster_ids.size, event_times.size, n_bins))

    for i_unit in numba.prange(len(cluster_ids)):
        unit_spike_times = spike_times[spike_cluster_ixs == i_unit]
        epoch_idxs = np.searchsorted(unit_spike_times, epoch_bounds)
        for i_event, (ep, t) in enumerate(zip(epoch_idxs, ts)):
            xind = (
                np.floor((unit_spike_times[ep[0] : ep[1]] - t[0]) / bin_size)
            ).astype(np.int64)
            r = np.bincount(xind, minlength=tscale.shape[0])
            binned_spikes[i_unit, i_event, :] = r[:-1]

    tscale = (tscale[:-1] + tscale[1:]) / 2

    return binned_spikes, tscale
