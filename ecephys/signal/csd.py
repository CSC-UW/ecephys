import numpy as np
import xarray as xr
from kcsd import KCSD1D


def get_kcsd(sig, ele_pos, drop_chans=[], do_lcurve=False, **kcsd_kwargs):
    """If signal units are in uV, then CSD units are in nA/mm."""
    channels = sig.channel.values  # Save for later
    sig = sig.assign_coords({"pos": ("channel", ele_pos)})
    sig = sig.drop_sel(channel=drop_chans, errors="ignore")
    k = KCSD1D(sig.pos.values.reshape(-1, 1), sig.values.T, **kcsd_kwargs)

    if do_lcurve:
        print("Performing L-curve parameter estimation...")
        k.L_curve()

    csd = xr.DataArray(
        k.values("CSD"),
        dims=("pos", "time"),
        coords={"pos": k.estm_x, "time": sig.time.values},
    )

    if (k.estm_x == ele_pos).all():
        csd = csd.assign_coords({"channel": ("pos", channels)})

    return csd.assign_attrs(kcsd=k, fs=sig.fs)


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