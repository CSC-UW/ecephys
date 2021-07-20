import numpy as np


def get_pitts_csd(trial_mean_lfp, spacing):
    """Compute current source density as the second spatial derivative along the probe
    length as a 1D approximation of the Laplacian, after Pitts (1952).


    Parameters
    ----------
    trial_mean_lfp: numpy.ndarray
        LFP traces surrounding presentation of a common stimulus that
        have been averaged over trials. Dimensions are channels X time samples.
        The input MUST be mean or median subtracted before use.
    spacing : float
        Distance between channels, in millimeters. This spacing may be
        physical distances between channels or a virtual distance if channels
        have been interpolated to new virtual positions.
    Returns
    -------
    Tuple[csd, csd_channels]:
        csd : numpy.ndarray
            Current source density. Dimensions are channels X time samples.
        csd_channels: numpy.ndarray
            Array of channel indices for CSD.

    References
    ----------
    [1] https://github.com/AllenInstitute/AllenSDK/blob/master/allensdk/brain_observatory/ecephys/current_source_density/_current_source_density.py
    """

    # Need to pad lfp channels for Laplacian approx.
    padded_lfp = np.pad(trial_mean_lfp, pad_width=((1, 1), (0, 0)), mode="edge")

    csd = (1 / (spacing ** 2)) * (
        padded_lfp[2:, :] - (2 * padded_lfp[1:-1, :]) + padded_lfp[:-2, :]
    )

    csd_channels = np.arange(0, trial_mean_lfp.shape[0])

    return (csd, csd_channels)