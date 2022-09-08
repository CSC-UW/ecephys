# -*- coding: utf-8 -*-
import numpy as np
import tqdm


def get_normed_data(data, normalize, window=None, binsize=None, verbose=True):
    '''Normalizes all PSTH data

    Args:
        data (2-D numpy array): Array with shape
            :data:`(n_neurons, n_bins)`
        normalize (str): If :data:`all`, divide all PSTHs by highest peak
            firing rate in all neurons. If :data:`each`, divide each PSTH
            by its own peak firing rate. If :data:`baseline_zscore`, zscore
            each PSTH relative to the baseline period (window < 0).
            If None, do not normalize.

    Returns:
        array: The original data array, normalized
    '''
    if not len(data):
        return data

    max_rates = np.amax(data, axis=1)

    # Computes the normalization factors.
    if normalize == 'all':
        if verbose:
            print("Normalize by all")
        norm_factors = np.ones([data.shape[0], 1]) * np.amax(max_rates)
        return np.divide(data, norm_factors, where=norm_factors!=0)
    elif normalize == 'each':
        if verbose:
            print("Normalize by each")
        norm_factors = (
            np.reshape(max_rates, (max_rates.shape[0], 1)) *
            np.ones((1, data.shape[1]))
        )
        return np.divide(data, norm_factors, where=norm_factors!=0)
    elif normalize is None:
        if verbose:
            print("Don't normalize")
        return data
    elif normalize == 'baseline_zscore':
        if verbose:
            print("Zscore by baseline")
        if window[0] > 0 or window[1] < 0:
            raise ValueError(f"Can't z-score by baseline for window {window}")
        # Baseline is bins < 0
        # n_baseline_idx = int(abs(window[0]) / binsize)
        n_baseline_idx = int(abs(window[0]) / binsize) - 1  # Exclude t=0 bin
        baseline_mean = np.tile(
            np.mean(data[:, 0:n_baseline_idx], axis=1),
            (data.shape[1], 1)
        ).transpose()  # nunits x nbins, repeat mean across 1st column
        baseline_std = np.tile(
            np.std(data[:, 0:n_baseline_idx], axis=1),
            (data.shape[1], 1)
        ).transpose()  # nunits x nbins, repeat std across 1st column
        # return (data - baseline_mean) / baseline_std
        # return data - baseline_mean
        res = np.zeros(data.shape)  # 0 if std = 0
        return np.divide(
            data - baseline_mean, baseline_std, where=baseline_std!=0, out=res
        )
    elif normalize == 'baseline_norm':
        if verbose:
            print("Normalize by baseline average (no z-score)")
        if window[0] > 0 or window[1] < 0:
            raise ValueError(f"Can't norm by baseline for window {window}")
        # Baseline is bins < 0
        n_baseline_idx = int(abs(window[0]) / binsize) - 1  # Exclude t=0 bin
        baseline_mean = np.tile(
            np.mean(data[:, 0:n_baseline_idx], axis=1),
            (data.shape[1], 1)
        ).transpose()  # nunits x nbins, repeat mean across 1st column
        res = np.zeros(data.shape)  # 0 if mean = 0
        return np.divide(
            data, baseline_mean, where=baseline_mean!=0, out=res
        )
    else:
        raise ValueError('Invalid value for `normalize` kwarg')


def get_evoked_firing_rates(spike_times, spike_clusters, unit_ids, events, plot_before, plot_after, time_bin):
    """
    Get event triggered firing rates

    Created on Fri Apr  2 11:11:28 2021

    @author: irene.rembado

    INPUTS:
        spike_times: array containing the time (in sec in the master/sync clock) of every spike detected in the whole recording across all channels 
        spike_clusters: array same length as spike_time which contains the cluster/unit identity for each of the spikes 
        unit_ids: sorted units
        events: array with time onsets of all the trials in seconds
        plot_before: time in second pre_event
        plot_after: time in second post_event
        time_bin: in second bin width to compile the PSTH
    
    OUTPUTS:
        evoked_firingrate = event triggered firing rate from the raster plot
        bins = array of times. it has 1 extra sample compared to evoked_firingrate
    """

    bins = np.arange(-plot_before, plot_after+time_bin, time_bin)
    evoked_firingrate = np.empty((len(unit_ids), len(bins)-1))*np.nan
    for indi, uniti in tqdm.tqdm(enumerate(unit_ids)):
        spikesi = np.squeeze(spike_times[spike_clusters == uniti])

        uniti_rates = []
        for E in events:
            window_spikes = spikesi[np.squeeze(np.argwhere((spikesi >= E-plot_before) & (spikesi <= E+plot_after)))]
            window_spikes = window_spikes - E
            sp_counts, edges = np.histogram(window_spikes, bins)
            uniti_rates.append(sp_counts/time_bin)

        uniti_rates = np.array(uniti_rates)
        evoked_firingrate[indi,:] = np.mean(uniti_rates, axis=0)

    return evoked_firingrate, bins