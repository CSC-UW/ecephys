from typing import Optional

import numba
import numba.typed
import numba.types
import numpy as np
import pandas as pd
import xarray as xr

from ecephys import hypnogram
from ecephys.units import binning, dtypes


def get_peths_from_trains(
    trains: dtypes.SpikeTrainDict_Secs,
    event_times: np.ndarray[np.float64],
    event_labels: Optional[np.ndarray] = None,
    pre_time: float = 0.4,
    post_time: float = 0.8,
    bin_size: float = 0.025,
    return_fr: bool = True,
    train_keys="spike_train",  # Could be acronym, etc. Special behavior if cluster_id
    property_frame: Optional[
        pd.DataFrame
    ] = None,  # Only applies if train_keys == "cluster_id"
    property_names: Optional[
        list[str]
    ] = None,  # Only applies if train_keys == "cluster_id"
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
    train_ids = np.array(list(trains.keys()))
    numba_trains = numba.typed.Dict.empty(numba.types.int64, numba.types.float64[:])
    for numba_id, train_id in enumerate(train_ids):
        numba_trains[numba_id] = trains[train_id]
    binned_spikes, tscale, numba_ids = _bin_spiketrains_numba(
        numba_trains, event_times, pre_time, post_time, bin_size
    )

    if return_fr:
        binned_spikes /= bin_size

    peths = xr.DataArray(
        binned_spikes,
        dims=(train_keys, "event", "time"),
        coords={
            train_keys: train_ids[np.array(numba_ids)],
            "event": event_times,
            "time": tscale,
        },
        attrs={"bin_size": bin_size},
    )

    if event_labels is not None:
        peths = peths.assign_coords({"event_type": ("event", event_labels)})

    if (train_keys == "cluster_id") and (property_frame is not None):
        peths = _add_cluster_properties_to_peths(peths, property_frame, property_names)

    return peths


# TODO: This should probably use and return an array of uint16, not float64.
@numba.njit(parallel=True, nogil=True, cache=True)
def _bin_spiketrains_numba(numba_trains, event_times, pre_time, post_time, bin_size):
    n_bins_pre = int(np.ceil(pre_time / bin_size))
    n_bins_post = int(np.ceil(post_time / bin_size))
    n_bins = n_bins_pre + n_bins_post
    tscale = np.arange(-n_bins_pre, n_bins_post + 1) * bin_size

    ts = np.repeat(event_times, tscale.size).reshape(-1, tscale.size) + tscale
    epoch_bounds = np.zeros(shape=(event_times.size, 2))
    epoch_bounds[:, 0] = ts[:, 0]
    epoch_bounds[:, 1] = ts[:, -1]

    train_ids = list(numba_trains.keys())
    binned_spikes = np.zeros(
        shape=(len(train_ids), event_times.size, n_bins), dtype=np.float64
    )
    for i_unit in numba.prange(len(train_ids)):
        id = train_ids[i_unit]
        unit_spike_times = numba_trains[id]
        epoch_idxs = np.searchsorted(unit_spike_times, epoch_bounds)
        for i_event, (ep, t) in enumerate(zip(epoch_idxs, ts)):
            xind = (
                np.floor((unit_spike_times[ep[0] : ep[1]] - t[0]) / bin_size)
            ).astype(np.int64)
            r = np.bincount(xind, minlength=tscale.shape[0])
            binned_spikes[i_unit, i_event, :] = r[:-1]

    tscale = (tscale[:-1] + tscale[1:]) / 2

    return binned_spikes, tscale, train_ids


def _add_cluster_properties_to_peths(
    peths: xr.DataArray,
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
        peths["cluster_id"].values
    ]  # Order cluster_ids (i.e. rows) of properties dataframe to match datarray order
    property_names = (
        property_frame.columns if property_names is None else property_names
    )
    coords = {col: ("cluster_id", property_frame[col].to_numpy()) for col in property_names}
    return peths.assign_coords(coords)


def get_peths_sem(
    peths: xr.DataArray, variance_dim: str = "event", group_variance_by: str = None
) -> xr.Dataset:
    """
    Get a standard error of the mean capturing variance across a chosen dimension for
    peri-event time histograms.

    Parameters
    ----------
    peths : xr.DataArray
        The peri-event time histogram.
    variance_dim : str, optional
        The dimension to calculate variance across.
    group_variance_by : str, optional
        The coordinate to group variance by. For example, if `variance_dim == "event"`
        and `group_variace_by == "state"`, the PETHs will be averaged across all events
        within each state, and variance across events WITHIN each state will be
        calculated.

    Returns
    -------
    xr.Dataset
        A dataset containing the SEM, mean, standard deviation, and number of events
        for each group.
    """
    if group_variance_by is not None:
        peths = peths.groupby(group_variance_by)
    mean_ = peths.mean(dim=variance_dim)
    std_ = peths.std(dim=variance_dim)
    n_ = peths.count(dim=variance_dim)
    sem_ = std_ / np.sqrt(n_)

    return xr.Dataset(
        {
            "mean": mean_,
            "std": std_,
            "n": n_,
            "sem": sem_,
        }
    )


def zscore_peths_by_peri_event_window(peths: xr.DataArray) -> xr.DataArray:
    """Z-score each individual PETH by its peri-event window. This is the preferred
    method for Z-scoring PETHs, especially when you want to compare across states.

    Parameters
    ----------
    peths : xr.DataArray
        The peri-event time histograms.

    Returns
    -------
    z_peths : xr.DataArray
        The z-scored peri-event time histograms.
    """
    mean_ = peths.mean(dim="time")
    std_ = peths.std(dim="time")
    return (peths - mean_) / std_


################
# Niche functions, you'll probably never use.
################


def bin_trains_and_group_clusters(
    cluster_trains: dict[str, np.ndarray],
    bin_size: float,
    group_clusters_by: str,
    grp2clus_map: dict[str, list[str]],
) -> xr.DataArray:
    groups = list(grp2clus_map.keys())

    bin_edges, t_min, t_max = binning.get_aligned_bins(cluster_trains, bin_size)
    binned_group_rates = np.zeros((len(groups), len(bin_edges) - 1))
    for i, (group, cluster_ids) in enumerate(grp2clus_map.items()):
        binned_cluster_rates = np.zeros((len(cluster_ids), len(bin_edges) - 1))
        for j, id in enumerate(cluster_ids):
            binned, _ = binning.bin_train(cluster_trains[id], bin_size, t_min, t_max)
            binned_cluster_rates[j, :] = binned
        binned_group_rates[i, :] = np.sum(binned_cluster_rates, axis=0)
    binned_group_rates = xr.DataArray(
        binned_group_rates,
        dims=[group_clusters_by, "time"],
        coords={group_clusters_by: groups, "time": bin_edges[:-1]},
    )
    binned_group_rates.attrs["bin_size"] = bin_size
    return binned_group_rates


def zscore_peths_by_whole_recording_state_specific_mean_and_std(
    evt_peths: xr.DataArray,
    spike_trains: dict[str, np.ndarray],
    bin_size: float,
    grp2clus_map: dict[str, np.ndarray],
    group_clusters_by: str,
    hg: hypnogram.Hypnogram,
) -> xr.DataArray:
    """
    Z-score peths by their whole-recording, state-specific mean and std.

    Because even the within-state firing rate statistics are non-stationary, this does
    not work as well as one might hope. It is better to z-score each event by the mean
    and std firing rates in a peri-event window specific to each event. This function
    is kept around as a reference, and to warn you against this method.

    Parameters
    ----------
    evt_peths : xr.DataArray
        The peths to z-score.
    spike_trains : dict[str, np.ndarray]
        The spike trains to use for z-scoring.
    bin_size : float
        The size of the time bins used for the peth.
    grp2clus_map : dict[str, np.ndarray]
        A dictionary mapping group names to cluster IDs.
        Usually obtained by:
        ```
        grp2clus_map = (
            mps.properties
            .groupby("acronym")["cluster_id"]
            .unique()
            .to_dict()
        )
        ```
    group_clusters_by : str
        Group by the clusters by this property. Usually "acronym".
    hg : ec.hypnogram.Hypnogram
        The hypnogram that will be used to assign state labels to the binned spike trains.
        Should be the same as the hypnogram used to compute the peths, ideally.

    Returns
    -------
    z_peths : xr.DataArray
        The z-scored peths.
    """
    # Bin whole-recording spike trains, and sum them within each group of clusters.
    binned_acronym_rates = bin_trains_and_group_clusters(
        spike_trains, bin_size, group_clusters_by, grp2clus_map
    )
    # Assign state labels to the binned spike trains.
    binned_acronym_rates = binned_acronym_rates.assign_coords(
        {"state": ("time", hg.get_states(binned_acronym_rates.time.values))}
    )

    # Compute state-specific mean and std of the binned spike trains.
    mean_ = binned_acronym_rates.groupby("state").mean(dim="time")
    std_ = binned_acronym_rates.groupby("state").std(dim="time")

    # Z-score each event by it's whole-recording, state-specific mean and std
    tmp_ = evt_peths.groupby(group_clusters_by).sum(dim="cluster_id")
    tmp_ = tmp_ - mean_.sel(state=tmp_.state)
    return tmp_ / std_.sel(state=tmp_.state)


def _get_peths_from_trains_alt(
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
        binned_spikes[i], tscale = _bin_single_spiketrain_numba(
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

    if event_labels is not None:
        peths = peths.assign_coords({"event_type": ("event", event_labels)})

    if property_frame is not None:
        peths = _add_cluster_properties_to_peths(peths, property_frame, property_names)

    return peths


@numba.jit(
    (numba.float64[:], numba.float64[:], numba.float64, numba.float64, numba.float64),
    nopython=True,
    nogil=True,
    cache=True,
)
def _bin_single_spiketrain_numba(
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


def _get_peths_from_spike_vector(
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
    binned_spikes, tscale = _bin_spike_vector_numba(
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

    if event_labels is not None:
        peths = peths.assign_coords({"event_type": ("event", event_labels)})

    if property_frame is not None:
        peths = _add_cluster_properties_to_peths(peths, property_frame, property_names)

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
def _bin_spike_vector_numba(
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
