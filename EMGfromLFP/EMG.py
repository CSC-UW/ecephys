# -*- coding: utf-8 -*-

"""
Derive EMG from LFP through correlation of highfrequency activity.

This code is basically a Python translation and readaptation of the
bz_EMGFromLFP.m function from Buzsaki's lab.
(`https://github.com/buzsakilab/buzcode/blob/master/detectors/bz_EMGFromLFP.m`).
Based on Erik Schomburg's and code and work, published in `Theta Phase
Segregation of Input-Specific Gamma Patterns in Entorhinal-Hippocampal Networks,
Schomburg et al., Neuron 2014.


Tom Bugnon, 01/2020
"""

import itertools

import numpy as np
import scipy.signal
import tqdm
from scipy.stats import pearsonr


def compute_EMG(lfp, sf, target_sf, window_size, bandpass, bandstop):
    """Derive EMG from LFP.

    Compute av. correlation across channel pairs in sliding windows.

    Args:
        lfp: (nchan x nsamples) lfp data array
        sf: Sampling frequency of input LFP array (Hz)
    Kwargs:
        target_sf: Sampling frequency of output "EMG" array (Hz)
        window_size: Duration (in ms) of each window during which average
            correlation is computed
        bandpass, bandstop: Used for bandpass filtering

    Returns:
        EMG_data: (1 x nSteps) data array. Sampling frequency is `targetsf`
        EMG_metadata: dictionary
    """

    print("Filtering LFP at bandpass={bandpass}, bandstop={bandstop}")
    lfp_filt = filter_data(lfp, bandpass, bandstop, sf)
    print("Computing EMG from filtered LFP...")
    print(f"target sf = {target_sf}, window size = {window_size}, LFP sf={sf},"
          f" LFP nchans = {lfp_filt.shape[0]}")
    EMG_data = compute_av_corr(lfp_filt, sf, target_sf, window_size)
    print('Done!')
    return EMG_data


def filter_data(data, bandpass, bandstop, sf):
    """Bandpass filter data along last dimension. """

    Wp = np.array(bandpass) / (sf/2)
    Ws = np.array(bandstop) / (sf/2)
    Rp = 3
    Rs = 20
    [N, Wn] = scipy.signal.cheb2ord(Wp, Ws, Rp, Rs)
    [b2, a2] = scipy.signal.cheby2(N, Rs, Wn, 'pass')

    return scipy.signal.filtfilt(b2, a2, data)


def compute_av_corr(data, data_sf, target_sf, window_size):
    """Compute av. correlation across channel pairs in sliding windows.

    Args:
        data: (nchan x nsamples) data array
        data_sf: Sampling frequency of data array (Hz)
    Kwargs:
        target_sf: Desired sampling frequency of output time course
            (1/windowStep) (Hz)
        window_size: Duration (in ms) of each window during which average
            correlation is computed

    Returns:
        corrData: (1 x nSteps) data array. Sampling frequency is `targetsf`
    """

    # Input data
    n_chans, n_samps = data.shape
    assert n_chans > 1

    # Number of samples in the original recording within each window
    window_n_samps = int(window_size * data_sf)
    # Timestamps of the center of each successive window
    tEnd = n_samps / data_sf
    tStep = 1 / target_sf
    win_timestamps = np.arange(0, tEnd, tStep)
    # Index in data of the samples closest to the center of each successive
    # window.
    window_center_samps = (win_timestamps * data_sf).astype(int)

    corr_data = np.zeros((1, len(window_center_samps)))
    chan_pairs = [
        (i, j) for i, j in itertools.product(range(n_chans), repeat=2)
        if i < j
    ]
    # Summate corrlation within each window for all channel pairs
    # Iterate on channel pairs, then on timestamps (C-style indexing)
    for i, j in tqdm.tqdm(chan_pairs, "XCorr: Iterate on channel pairs"):
        for s, win_center_samp in enumerate(window_center_samps):
            # Start and end samples for that window. First and last windows are
            # shorter
            win_start_i = max(0, int(win_center_samp - window_n_samps/2))
            win_end_i = min(n_samps, int(win_center_samp + window_n_samps/2))
            # Add correlation for pair to total
            corr_data[0, s] += pearsonr(
                data[i, win_start_i:win_end_i+1],
                data[j, win_start_i:win_end_i+1],
            )[0]

    # Normalize after summation
    corr_data /= len(chan_pairs)

    return corr_data
