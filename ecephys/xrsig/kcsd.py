import xarray as xr
import numpy as np
from kcsd import KCSD1D


def get_kcsd(sig, ele_pos, drop_chans=[], do_lcurve=False, **kcsd_kwargs):
    """If signal units are in uV, then CSD units are in nA/mm.

    Paramters:
    ----------
    sig: xr.DataArray (time, channel)
        The data on which to compute a 1D KCSD.
    ele_pos: (n_channels,)
        The positions, in mm, of each electrode in `sig`.
    drop_chans: list
        Channels (as they appear in `sig`) to exclude when estimating the CSD.
    do_lcurve: Boolean
        Whether to perform L-Curve parameter estimation.
    **kcsd_kwargs:
        Keywords passed to KCSD1D.

    Returns:
    --------
    csd: xr.DataArray (time, pos)
        The CSD estimates. If the estimation locations requested of KCSD1D correspond
        exactly to electrode positions, a `channel` coordinate on the `pos` dimension
        will give corresponding channels for each estimate.
    """
    channels = sig.channel.values  # Save for later
    sig = sig.assign_coords({"pos": ("channel", ele_pos)})
    sig = sig.drop_sel(channel=drop_chans, errors="ignore")
    k = KCSD1D(
        sig.pos.values.reshape(-1, 1),
        sig.transpose("channel", "time").values,
        **kcsd_kwargs
    )

    if do_lcurve:
        print("Performing L-curve parameter estimation...")
        k.L_curve()

    csd = xr.DataArray(
        k.values("CSD"),
        dims=("pos", "time"),
        coords={"pos": k.estm_x, "time": sig.time.values},
    )

    if np.allclose(k.estm_x, ele_pos):
        csd = csd.assign_coords({"channel": ("pos", channels)})

    return csd.assign_attrs(kcsd=k, fs=sig.fs)