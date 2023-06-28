import math

import numba
import numpy as np
import numpy.typing as npt
import pandas as pd
import spikeinterface.postprocessing.correlograms
import xarray as xr

from ecephys import hypnogram
from ecephys.units import cluster_trains
from ecephys.units import dtypes


def validate_cluster_correlograms(da: dtypes.XArray):
    dims = ("clusterA", "clusterB", "time")
    if not tuple(da.dims) == dims:
        raise AttributeError(
            f"Cluster correlogram DataArray must have dimensions {dims})"
        )


def compute_intrapopulation_correlograms(
    spike_times: dtypes.SpikeTrain_Secs,
    spike_cluster_ixs: dtypes.ClusterIXs,
    cluster_ids: dtypes.ClusterIDs,
    window_ms: float = 50.0,
    bin_ms: float = 1.0,
    do_autocorr: bool = True,
) -> xr.DataArray:
    bins, half_window_size, bin_size = make_bins(window_ms, bin_ms)
    num_bins = 2 * int(half_window_size / bin_size)
    num_units = len(cluster_ids)
    correlograms = np.zeros((num_units, num_units, num_bins), dtype=np.int64)
    _compute_intrapopulation_time_correlograms_numba(
        correlograms,
        spike_times,
        spike_cluster_ixs.astype(np.int32),
        half_window_size,
        bin_size,
        do_autocorr,
    )
    return xr.DataArray(
        correlograms.transpose(
            1, 0, 2
        ),  # Transpose, so that [A, B, :] is the histogram of (spiketimes(B) - spiketimes(A))
        dims=("clusterA", "clusterB", "time"),
        coords={
            "clusterA": cluster_ids,
            "clusterB": cluster_ids,
            "time": bins[:-1],
        },
        attrs={"window_ms": window_ms, "bin_ms": bin_ms},
    )


def compute_interpopulation_correlograms(
    spike_times: dtypes.SpikeTrain_Secs,
    spike_cluster_ixs: dtypes.ClusterIXs,
    cluster_ids: dtypes.ClusterIDs,
    pop_a_ids: dtypes.ClusterIDs,
    pop_b_ids: dtypes.ClusterIDs,
    window_ms: float = 50.0,
    bin_ms: float = 1.0,
) -> xr.DataArray:
    bins, half_window_size, bin_size = make_bins(window_ms, bin_ms)
    num_bins = 2 * int(half_window_size / bin_size)
    num_units = len(cluster_ids)
    correlograms = np.zeros((num_units, num_units, num_bins), dtype=np.int64)
    pop_a_ixs = np.where(np.isin(cluster_ids, pop_a_ids))[0]
    pop_b_ixs = np.where(np.isin(cluster_ids, pop_b_ids))[0]
    _compute_interpopulation_time_correlograms_numba(
        correlograms,
        spike_times,
        spike_cluster_ixs.astype(np.int32),
        pop_a_ixs.astype(np.int32),
        pop_b_ixs.astype(np.int32),
        half_window_size,
        bin_size,
    )
    return xr.DataArray(
        correlograms.transpose(
            1, 0, 2
        ),  # Transpose, so that [A, B, :] is the histogram of (spiketimes(B) - spiketimes(A))
        dims=("clusterA", "clusterB", "time"),
        coords={
            "clusterA": cluster_ids,
            "clusterB": cluster_ids,
            "time": bins[:-1],
        },
        attrs={"window_ms": window_ms, "bin_ms": bin_ms},
    )


def compute_intrapopulation_correlograms_by_spike_type(
    spike_times: dtypes.SpikeTrain_Secs,
    spike_cluster_ixs: dtypes.ClusterIXs,
    spike_types: np.ndarray,
    cluster_ids: dtypes.ClusterIDs,
    window_ms: float = 50,
    bin_ms: float = 1,
    do_autocorr: bool = True,
) -> xr.Dataset:
    correlograms = {}
    for type_ in np.unique(spike_types):
        is_type_ = spike_types == type_
        correlograms[type_] = compute_intrapopulation_correlograms(
            spike_times[is_type_],
            spike_cluster_ixs[is_type_],
            cluster_ids,
            window_ms,
            bin_ms,
            do_autocorr,
        )

    return xr.Dataset(correlograms)


def compute_intrapopulation_correlograms_by_hypnogram_state(
    trains: dtypes.ClusterTrains_Secs,
    hg: hypnogram.FloatHypnogram,
    states: list[str] = None,
    window_ms: float = 50,
    bin_ms: float = 1,
    do_autocorr: bool = True,
) -> xr.Dataset:
    """
    If you have a hypnogram, use compute_intrapopulation_correlograms_by_hypnogram_state!
    It is much faster than compute_intrapopulation_correlograms_by_spike_type! 30s vs 3min!
    """
    trains_by_state = {}
    states = hg["state"].unique() if states is None else states
    for state in states:
        state_hg = hg.keep_states([state])
        trains_by_state[state] = {
            id: tr[state_hg.covers_time(tr)] for id, tr in trains.items()
        }

    correlograms = {}
    for state, state_trains in trains_by_state.items():
        (
            spike_times,
            spike_cluster_ixs,
            cluster_ids,
        ) = cluster_trains.convert_cluster_trains_to_spike_vector(state_trains)
        correlograms[state] = compute_intrapopulation_correlograms(
            spike_times, spike_cluster_ixs, cluster_ids, window_ms, bin_ms, do_autocorr
        )

    return xr.Dataset(correlograms)


def compute_interpopulation_correlograms_by_spike_type(
    spike_times: dtypes.SpikeTrain_Secs,
    spike_cluster_ixs: dtypes.ClusterIXs,
    spike_types: np.ndarray,
    cluster_ids: dtypes.ClusterIDs,
    pop_a_ids: dtypes.ClusterIDs,
    pop_b_ids: dtypes.ClusterIDs,
    window_ms: float = 50,
    bin_ms: float = 1,
) -> xr.Dataset:
    """
    If you have a hypnogram, use compute_interpopulation_correlograms_by_hypnogram_state!
    It is much faster than compute_interpopulation_correlograms_by_spike_type! 30s vs 3min!
    """
    correlograms = {}
    for type_ in np.unique(spike_types):
        is_type_ = spike_types == type_
        correlograms[type_] = compute_interpopulation_correlograms(
            spike_times[is_type_],
            spike_cluster_ixs[is_type_],
            cluster_ids,
            pop_a_ids,
            pop_b_ids,
            window_ms,
            bin_ms,
        )

    return xr.Dataset(correlograms)


def compute_interpopulation_correlograms_by_hypnogram_state(
    trains: dtypes.ClusterTrains_Secs,
    hg: hypnogram.FloatHypnogram,
    pop_a_ids: dtypes.ClusterIDs,
    pop_b_ids: dtypes.ClusterIDs,
    states: list[str] = None,
    window_ms: float = 50,
    bin_ms: float = 1,
) -> xr.Dataset:
    """
    If you have a hypnogram, use compute_interpopulation_correlograms_by_hypnogram_state!
    It is much faster than compute_interpopulation_correlograms_by_spike_type! 30s vs 3min!
    """
    trains_by_state = {}
    states = hg["state"].unique() if states is None else states
    for state in states:
        state_hg = hg.keep_states([state])
        trains_by_state[state] = {
            id: tr[state_hg.covers_time(tr)] for id, tr in trains.items()
        }

    correlograms = {}
    for state, state_trains in trains_by_state.items():
        (
            spike_times,
            spike_cluster_ixs,
            cluster_ids,
        ) = cluster_trains.convert_cluster_trains_to_spike_vector(state_trains)
        correlograms[state] = compute_interpopulation_correlograms(
            spike_times,
            spike_cluster_ixs,
            cluster_ids,
            pop_a_ids,
            pop_b_ids,
            window_ms,
            bin_ms,
        )

    return xr.Dataset(correlograms)


def make_bins(
    window_ms: float, bin_ms: float
) -> tuple[npt.NDArray[np.float64], float, float]:
    """Make correlogram bins."""
    # CAUTION: window_size is HALF window_ms. This is so that t=0 is always the left edge of a bin
    half_window_size = window_ms / 2 * 1e-3
    bin_size = bin_ms * 1e-3
    num_bins = 2 * int(half_window_size / bin_size)
    assert num_bins >= 2, "Requested correlogram would produce < 2 bins"
    assert np.isclose(
        num_bins * bin_size, half_window_size * 2
    ), "Requested correlogram would not be 0-centered"

    # np.arange is numerically unstable, so use linspace instead
    bins, step = np.linspace(
        -half_window_size,
        half_window_size + bin_size,
        num_bins + 1,
        endpoint=False,
        retstep=True,
    )
    assert np.isclose(step, bin_size), "Bin size is not as expected"

    return bins, half_window_size, bin_size


@numba.jit(
    (numba.float64[::1], numba.float32, numba.float32),
    nopython=True,
    nogil=True,
    cache=True,
)
def _compute_time_autocorr_numba(spike_times, half_window_size, bin_size):
    num_half_bins = int(half_window_size / bin_size)
    num_bins = 2 * num_half_bins

    auto_corr = np.zeros(num_bins, dtype=np.int64)

    for i in range(len(spike_times)):
        for j in range(i + 1, len(spike_times)):
            diff = spike_times[j] - spike_times[i]

            if diff > half_window_size:
                break

            bin = int(math.floor(diff / bin_size))
            auto_corr[num_half_bins + bin] += 1

            bin = int(math.floor(-diff / bin_size))
            auto_corr[num_half_bins + bin] += 1

    return auto_corr


@numba.jit(
    (numba.float64[::1], numba.float64[::1], numba.float32, numba.float32),
    nopython=True,
    nogil=True,
    cache=True,
)
def _compute_time_crosscorr_numba(
    spike_times1, spike_times2, half_window_size, bin_size
):
    num_half_bins = int(half_window_size / bin_size)
    num_bins = 2 * num_half_bins

    cross_corr = np.zeros(num_bins, dtype=np.int64)

    start_j = 0
    for i in range(len(spike_times1)):
        for j in range(start_j, len(spike_times2)):
            diff = spike_times1[i] - spike_times2[j]

            if diff >= half_window_size:
                start_j += 1
                continue
            if diff < -half_window_size:
                break

            bin = int(math.floor(diff / bin_size))
            cross_corr[num_half_bins + bin] += 1

    return cross_corr


@numba.jit(
    (
        numba.int64[:, :, ::1],
        numba.float64[::1],
        numba.int32[::1],
        numba.float32,
        numba.float32,
        numba.boolean,
    ),
    nopython=True,
    nogil=True,
    cache=True,
    parallel=True,
)
def _compute_intrapopulation_time_correlograms_numba(
    correlograms, spike_times, spike_labels, half_window_size, bin_size, do_autocorr
):
    n_units = correlograms.shape[0]

    for i in numba.prange(n_units):
        # ~ for i in range(n_units):
        spike_times1 = spike_times[spike_labels == i]

        for j in range(i, n_units):
            spike_times2 = spike_times[spike_labels == j]

            if do_autocorr and (i == j):
                correlograms[i, j, :] += _compute_time_autocorr_numba(
                    spike_times1, half_window_size, bin_size
                )
            else:
                cc = _compute_time_crosscorr_numba(
                    spike_times1, spike_times2, half_window_size, bin_size
                )
                correlograms[i, j, :] += cc
                correlograms[j, i, :] += cc[::-1]


@numba.jit(
    (
        numba.int64[:, :, ::1],
        numba.float64[::1],
        numba.int32[::1],
        numba.int32[::1],
        numba.int32[::1],
        numba.float32,
        numba.float32,
    ),
    nopython=True,
    nogil=True,
    cache=True,
    parallel=True,
)
def _compute_interpopulation_time_correlograms_numba(
    correlograms,
    spike_times,
    spike_cluster_ixs,
    pop_a_cluster_ixs,
    pop_b_cluster_ixs,
    half_window_size,
    bin_size,
):
    for a in numba.prange(pop_a_cluster_ixs.size):
        ix_a = pop_a_cluster_ixs[a]
        spike_times_a = spike_times[spike_cluster_ixs == ix_a]

        for b in range(pop_b_cluster_ixs.size):
            ix_b = pop_b_cluster_ixs[b]
            spike_times_b = spike_times[spike_cluster_ixs == ix_b]

            cc = _compute_time_crosscorr_numba(
                spike_times_a, spike_times_b, half_window_size, bin_size
            )
            correlograms[ix_a, ix_b, :] += cc
            correlograms[ix_b, ix_a, :] += cc[::-1]


def add_cluster_properties_to_correlograms(
    correlogram: dtypes.XArray,
    properties_frame: pd.DataFrame,  # The dataframe containing the property information
    property_names: list[
        str
    ] = None,  # Properties to add as coordinates. If None, add all properties.
) -> dtypes.XArray:
    """Assign coordinates representing each cluster's properties."""
    validate_cluster_correlograms(correlogram)

    if property_names is None:
        property_names = [
            prop for prop in properties_frame.columns if prop != "cluster_id"
        ]

    _properties_frame = properties_frame.set_index("cluster_id").loc[
        correlogram["clusterA"].values
    ]  # Order cluster_ids (i.e. rows) of properties dataframe to match datarray order
    coords = {
        f"{prop}A": ("clusterA", _properties_frame[prop].values)
        for prop in property_names
    }
    correlogram = correlogram.assign_coords(coords)

    _properties_frame = properties_frame.set_index("cluster_id").loc[
        correlogram["clusterB"].values
    ]  # Order cluster_ids (i.e. rows) of properties dataframe to match datarray order
    coords = {
        f"{prop}B": ("clusterB", _properties_frame[prop].values)
        for prop in property_names
    }
    correlogram = correlogram.assign_coords(coords)

    return correlogram


#####
# Functions for computing correlgorams from spike frames, instead of spike times
# Might be faster than using times? Untested.
#####


def _make_frame_bins(
    fs: float, window_ms: float, bin_ms: float
) -> tuple[npt.NDArray[np.float64], int, int]:
    """Make correlogram bins.
    Modified from spikeinterface.postprocessing.correlograms._make_bins.
    Changed to take fs instead of sorting.
    """
    half_window_size = int(round(fs * window_ms / 2 * 1e-3))  # In samples.
    bin_size = int(round(fs * bin_ms * 1e-3))
    half_window_size -= half_window_size % bin_size
    num_bins = 2 * int(half_window_size / bin_size)
    assert num_bins >= 1

    bins = (
        np.arange(-half_window_size, half_window_size + bin_size, bin_size) * 1e3 / fs
    )

    return bins, half_window_size, bin_size


def _compute_simple_frame_correlograms(
    spike_frames: dtypes.SpikeTrain_Samples,
    spike_cluster_ixs: dtypes.ClusterIXs,
    cluster_ids: dtypes.ClusterIDs,
    fs: float,
    window_ms: float = 50,
    bin_ms: float = 1,
) -> xr.DataArray:
    bins, half_window_size, bin_size = _make_frame_bins(fs, window_ms, bin_ms)
    num_bins = 2 * int(half_window_size / bin_size)
    num_units = len(cluster_ids)
    correlograms = np.zeros((num_units, num_units, num_bins), dtype=np.int64)
    spikeinterface.postprocessing.correlograms._compute_correlograms_numba(
        correlograms,
        spike_frames.astype(np.int64),
        spike_cluster_ixs.astype(np.int32),
        half_window_size,
        bin_size,
    )
    return xr.DataArray(
        correlograms.transpose(
            1, 0, 2
        ),  # Transpose, so that [A, B, :] is the histogram of (spiketimes(B) - spiketimes(A))
        dims=("clusterA", "clusterB", "time"),
        coords={
            "clusterA": cluster_ids,
            "clusterB": cluster_ids,
            "time": bins[:-1] / 1000,  # Convert from msec to sec
        },
        attrs={"window_ms": window_ms, "bin_ms": bin_ms},
    )


def _compute_frame_correlograms_by_spike_type(
    spike_frames: dtypes.SpikeTrain_Samples,
    spike_cluster_ixs: dtypes.ClusterIXs,
    spike_types: np.ndarray,
    cluster_ids: dtypes.ClusterIDs,
    fs: float,
    window_ms: float = 50,
    bin_ms: float = 1,
) -> xr.Dataset:
    correlograms = {}
    for type_ in np.unique(spike_types):
        is_type_ = spike_types == type_
        correlograms[type_] = _compute_simple_frame_correlograms(
            spike_frames[is_type_],
            spike_cluster_ixs[is_type_],
            cluster_ids,
            fs,
            window_ms,
            bin_ms,
        )

    return xr.Dataset(correlograms)
