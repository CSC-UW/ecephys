import numpy as np


def jitter_train(train, bin_size, n_surrogates=1):
    """Jitter a spike train by splitting it into windows (bins), and within each window,
    randomly jittering each spike to a new time within the window, preserving the
    overall spike count in each window. This destroys precise synchrony on timescales
    below the window size, but keeps slow rate fluctuations intact.

    Can be used for any event type, not just spikes.

    Adapted from elephant.spike_train_surrogates.jitter_spikes().
    Their spike trains are assumed to start at t=0, ours are not. Hence the changes.

    Parameters
    ----------
    train : np.ndarray
        A 1D array of spike times.
    bin_size : float
        The size of the bins to split the spike train into. Same unit of time as train.
    n_surrogates : int, optional
        The number of surrogates to generate.

    Returns
    -------
    surrogates : np.ndarray
        A 2D array of surrogate spike trains. Shape is (n_surrogates, len(train)).
    """
    t_min = train.min()
    t_max = train.max()

    # Compute (right) bin edges for the jittering procedure
    # !: the last bin arrives until spiketrain.t_stop and might have size != bin_size
    bin_edges = np.arange(t_min, t_max, bin_size)
    bin_edges = np.hstack([bin_edges, t_max])

    # Create n surrogates with spikes randomly placed in the interval (0,1)
    surrogates = np.random.random_sample((n_surrogates, len(train)))

    # Compute the bin id of each spike
    bin_ids = np.array(((train - t_min) / bin_size), dtype=int)

    # Compute the size of each time bin (as a numpy array)
    bin_sizes = np.diff(bin_edges)

    # For each spike compute its offset (the left end of the bin it falls
    # into) and the size of the bin it falls into
    offsets = np.array([bin_edges[bin_id] for bin_id in bin_ids])
    dilats = np.array([bin_sizes[bin_id] for bin_id in bin_ids])

    # Compute each surrogate by dilating and shifting each spike s in the
    # poisson 0-1 spike trains to dilat * s + offset. Attach time unit again
    return np.sort(surrogates * dilats + offsets, axis=1)
