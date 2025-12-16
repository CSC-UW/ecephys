import numpy as np

from ecephys.units import dtypes


def bin_train(
    train: dtypes.SpikeTrain, bin_size: float, t_min: float = None, t_max: float = None
) -> tuple[dtypes.SpikeTrain, np.ndarray]:
    """
    Bins a spike train into a set of time bins and returns the rate in each bin.

    Parameters
    ----------
    train : dtypes.SpikeTrain
        The spike train to bin.
    bin_size : float
        The size of the time bins.
    t_min : float, optional
        The minimum time to bin. If None, the minimum time of the spike train is used.
    t_max : float, optional
        The maximum time to bin. If None, the maximum time of the spike train is used.

    All times (train event times, bin_size, t_min, t_max) should be in the same units.

    Returns
    -------
    np.ndarray
        The rate in each bin.
    np.ndarray
        The bin_edges, left-closed, right-open.
    """
    if t_min is None:
        t_min = train.min()
    if t_max is None:
        t_max = train.max()

    # Compute left-closed, right-open bin edges for the jittering procedure
    # !: the last bin arrives until spiketrain.t_stop and might have size != bin_size
    bin_edges = np.arange(t_min, t_max, bin_size)
    if bin_edges[-1] < t_max:
        bin_edges = np.hstack([bin_edges, t_max])

    # Compute the bin id of each spike
    bin_ids = np.array(((train - t_min) / bin_size), dtype=int)

    # Compute the size of each time bin (as a numpy array)
    bin_sizes = np.diff(bin_edges)

    # Count the number of spikes in each bin
    bin_counts = np.bincount(bin_ids, minlength=len(bin_sizes))

    # Get the rate in each bin
    bin_rates = bin_counts / bin_sizes

    return bin_rates, bin_edges


def get_aligned_bins(
    trains: dtypes.SpikeTrainDict, bin_size: float
) -> tuple[np.ndarray, float, float]:
    """Get a single set of bin edges for a set of spike trains, ensuring that the
    resulting bins will encompass all spikes.
    """
    t_min = np.min([np.min(tr) for tr in trains.values()])
    t_max = np.max([np.max(tr) for tr in trains.values()])
    first_train = next(iter(trains.values()))
    _, bin_edges = bin_train(first_train, bin_size, t_min, t_max)
    return bin_edges, t_min, t_max


def bin_trains(
    trains: dtypes.SpikeTrainDict, bin_size: float
) -> tuple[dtypes.SpikeTrainDict, np.ndarray]:
    """Bin spike trains, ensuring that all trains are binned to the same bin edges.

    Parameters
    ----------
    trains : dtypes.SpikeTrainDict
        A dictionary of spike trains.
    bin_size : float
        The size of the time bins.

    Returns
    -------
    binned_trains : dtypes.SpikeTrainDict
        A dictionary of binned rates.
    bin_edges : np.ndarray
        The bin edges.
    """
    bin_edges, t_min, t_max = get_aligned_bins(trains, bin_size)
    binned_trains = {
        id: bin_train(tr, bin_size, t_min, t_max)[0] for id, tr in trains.items()
    }
    return binned_trains, bin_edges
