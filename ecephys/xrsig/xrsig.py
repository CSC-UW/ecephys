import logging

import kcsd
import neuropixel
from neurodsp import fourier
from neurodsp import voltage
import neurodsp.utils
import numpy as np
from tqdm.auto import tqdm
import xarray as xr

import ecephys.emg_from_lfp
import ecephys.signal as npsig
from ecephys import utils

logger = logging.getLogger(__name__)


def validate_2d_timeseries(
    da: xr.DataArray,
    timedim: str = "time",
    sigdim: str = "channel",
    check_times: bool = False,
):
    if not da.dims == (timedim, sigdim):
        raise AttributeError(
            f"Timeseries2D DataArray must have dimensions ({timedim}, {sigdim})"
        )
    if not "fs" in da.attrs:
        raise ValueError("Timeseries2D must have sampling rate attr `fs`")
    if check_times and not np.all(np.diff(da[timedim].values) >= 0):
        raise ValueError("Timeseries2D times must be monotonically increasing.")


def validate_laminar(da: xr.DataArray, sigdim: str = "channel", lamdim: str = "y"):
    if not sigdim in da.dims:
        raise AttributeError(f"Laminar DataArray must include a channel dimension.")
    if not lamdim in da[sigdim].coords:
        raise AttributeError(
            f"Laminar DataArray must have {lamdim} coordinate on {sigdim} dimension."
        )


def get_pitch(da: xr.DataArray) -> xr.DataArray:
    """Get the vertical spacing between electrode sites, in microns"""
    validate_laminar(da)
    vals = np.diff(np.unique(da["y"].values))
    assert utils.all_equal(vals), f"Electrode pitch is not uniform. Pitches:\n {vals}"
    return np.absolute(vals[0])


def decimate_timeseries(da: xr.DataArray, q: int) -> xr.DataArray:
    validate_2d_timeseries(da)
    dat = npsig.decimate_timeseries(da, q=q)
    res = xr.DataArray(
        dat,
        dims=da.dims,
        coords={**da["time"][::q].coords, **da["channel"].coords},
        attrs=da.attrs,
    )
    res.attrs["fs"] = da.fs / q
    return res


def spatially_interpolate_timeseries(
    da: xr.DataArray,
    interp_me: list,  # The channels that should be interpolated
    inplace: bool = True,
) -> xr.DataArray:
    validate_2d_timeseries(da)
    validate_laminar(da)
    do_interp = np.isin(da["channel"], interp_me)
    if not do_interp.any():
        logger.debug(
            "None of the requested signals are present in the data. Doing nothing."
        )
        return

    # Get x coordinates if available, otherwise assume that channels are colinear
    if "x" in da["channel"].coords:
        x = da["x"].values
    else:
        print(
            "Data do not contain x coordinates on channel dimension. Assuming all electrodes are colinear."
        )
        x = np.zeros_like(da["y"])

    if inplace:
        da = da.copy()
    da.values = voltage.interpolate_bad_channels(
        lf.values.T, do_interp.astype("int"), x=x, y=da["y"].values
    ).T
    return da


def dephase_neuropixels(
    da: xr.DataArray, q: int = 1, inplace: bool = True
) -> xr.DataArray:
    validate_2d_timeseries(da)
    hdr = neuropixel.trace_header(version=1)
    shifts = hdr["sample_shift"][da["channel"].values] / q
    if not inplace:
        da = da.copy()
    da.values = fourier.fshift(da.values, shifts, axis=0)
    return da


def preprocess_neuropixels_ibl_style(
    da: xr.DataArray,
    bad_chans: list,
    downsample_factor: int = 4,
    chunk_size: int = 2**16,
    chunk_overlap: int = 2**10,
) -> xr.DataArray:
    validate_2d_timeseries(da)
    validate_laminar(da)
    wg = neurodsp.utils.WindowGenerator(
        ns=da["time"].size, nswin=chunk_size, overlap=chunk_overlap
    )
    segments = list()
    for first, last in tqdm(list(wg.firstlast)):
        seg = da.isel(time=slice(first, last))
        seg = decimate_timeseries(seg, q=downsample_factor)
        seg = dephase_neuropixels(seg)  # Shouldn't q be passed again here?
        seg = spatially_interpolate_timeseries(seg, bad_chans)
        first_valid = 0 if first == 0 else int(wg.overlap / 2 / downsample_factor)
        last_valid = (
            seg["time"].size
            if last == da["time"].size
            else int(seg["time"].size - wg.overlap / 2 / downsample_factor)
        )
        segments.append(seg.isel(time=slice(first_valid, last_valid)))

    return xr.concat(segments, dim="time")


def get_synthetic_emg_defaults() -> dict:
    return dict(
        target_sf=20,
        window_size=25.0,
        wp=[300, 600],
        ws=[275, 625],
        gpass=1,
        gstop=60,
        ftype="butter",
    )


def synthetic_emg(da: xr.DataArray, emg_kwargs: dict = None):
    """Estimate the EMG from LFP signals, using the `emg_from_lfp` subpackage.

    Parameters used for the computation are stored as attributes on the returned DataArray.

    Parameters:
    -----------
    **emg_kwargs:
        Keyword arguments passed to `emg_from_lfp.compute_emg()`

    Returns:
    --------
    DataArray:
        EMG with time dimension and timedelta, datetime coords.
    """
    validate_2d_timeseries(da)
    defaults = get_synthetic_emg_defaults()
    emg_kwargs = defaults if emg_kwargs is None else defaults.update(emg_kwargs)
    assert da.fs > (
        emg_kwargs["ws"][-1] * 2
    ), "EMG computation will fail trying to filter above the Nyquest frequency"

    emg_values = ecephys.emg_from_lfp.compute_emg(
        da.values.T, da.fs, **emg_kwargs
    ).flatten()
    emg_times = np.linspace(da["time"].min(), da["time"].max(), emg_values.size)
    emg = xr.DataArray(
        emg_values,
        dims="time",
        coords={
            "time": emg_times,
        },
        attrs={"units": "corr"},
    )
    for key, val in emg_kwargs.items():
        emg.attrs[key] = emg_kwargs[key]
    return emg


def kernel_current_source_density(
    da: xr.DataArray, drop=[], do_lcurve=False, **kcsd_kwargs
) -> xr.DataArray:
    """Compute 1D kernel current source density.
    If signal units are in uV, then CSD units are in nA/mm.

    Required coords:
    ----------------
    y, with units in um

    Paramters:
    ----------
    drop_chans: list
        Channels (as they appear in `self`) to exclude when estimating the CSD.
    do_lcurve: Boolean
        Whether to perform L-Curve parameter estimation.
    **kcsd_kwargs:
        Keywords passed to KCSD1D.

    Returns:
    --------
    csd: KernelCurrentSourceDensity
        The CSD estimates. If the estimation locations requested of KCSD1D correspond
        exactly to electrode positions, a `channel` coordinate on the `pos` dimension
        will give corresponding channels for each estimate.
    """
    validate_2d_timeseries(da)
    validate_laminar(da)
    umPerMm = 1000

    # Make sure we get CSD estimates at electrode locations, rather than say, in between electrodes.
    pitch_mm = get_pitch(da) / umPerMm  # Convert um to mm for KCSD package.
    gdx = kcsd_kwargs.get("gdx", None)
    if (gdx is not None) and (gdx != pitch_mm):
        raise ValueError("Requested gdx does not match electrode pitch.")
    else:
        kcsd_kwargs.update(gdx=pitch_mm)

    # Drop bad signals and redundant signals
    good = da.drop_sel({"channel": drop}, errors="ignore")
    u, ix = np.unique(good["y"], return_index=True)
    good = good.isel({"channel": ix})

    # Convert um to mm for KCSD package.
    elePosMm = good["y"].values / umPerMm

    # Compute kCSD
    k = kcsd.KCSD1D(
        elePosMm.reshape(-1, 1),
        good.transpose("channel", "time").values,
        **kcsd_kwargs,
    )
    if do_lcurve:
        print("Performing L-Curve parameter estimation...")
        k.L_curve()

    # Check and format result
    estm_locs = np.round(k.estm_x * umPerMm)
    mask = da["y"].isin(estm_locs)
    assert (
        estm_locs.size == mask.sum()
    ), "CSD returned estimates that do not match original signal positions exactly."
    csd = xr.zeros_like(da.sel({"channel": mask}))
    csd.values = k.values("CSD").T
    return csd.assign_attrs(kcsd=k)
