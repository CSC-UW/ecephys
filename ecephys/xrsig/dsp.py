import xarray as xr
import pandas as pd
import numpy as np
from ..signal import timefrequency as tfr
from .xrsig import get_dim_coords

# TODO: Checkut xrft package!


def _wrap_spg_times(spg_times, sig):
    time = sig.time.values.min() + spg_times
    timedelta = sig.timedelta.values.min() + pd.to_timedelta(spg_times, "s")
    datetime = sig.datetime.values.min() + pd.to_timedelta(spg_times, "s")
    return time, timedelta, datetime


def _wrap_2d_spectrogram(freqs, spg_time, spg, sig):
    time, timedelta, datetime = _wrap_spg_times(spg_time, sig)
    da = xr.DataArray(
        np.atleast_2d(spg),
        dims=("frequency", "time"),
        coords={
            "frequency": freqs,
            "time": time,
            "timedelta": ("time", timedelta),
            "datetime": ("time", datetime),
        },
    ).assign_coords(get_dim_coords(sig, None))
    da = da.assign_attrs(sig.attrs)
    if "units" in da.attrs:
        da = da.assign_attrs({"units": f"{da.units}^2/Hz"})
    return da


def _wrap_3d_spectrogram(freqs, spg_time, spg, sig):
    time, timedelta, datetime = _wrap_spg_times(spg_time, sig)
    da = xr.DataArray(
        np.atleast_3d(spg),
        dims=("frequency", "time", "channel"),
        coords={
            "frequency": freqs,
            "time": time,
            "timedelta": ("time", timedelta),
            "datetime": ("time", datetime),
        },
    ).assign_coords(get_dim_coords(sig, "channel"))
    da = da.assign_attrs(sig.attrs)
    if "units" in da.attrs:
        da = da.assign_attrs({"units": f"{da.units}^2/Hz"})
    return da


def parallel_spectrogram_welch(sig, **kwargs):
    """Compute a spectrogram for each channel in parallel.

    Parameters
    ----------
    sig: xr.DataArray (time, channel)
        Required attr: (fs, the sampling rate)
    **kwargs: optional
        Keyword arguments passed to `_compute_spectrogram_welch`.

    Returns:
    --------
    spg : xr.DataArray (frequency, time, channel)
        Spectrogram of `sig`.
    """
    freqs, spg_time, spg = tfr.parallel_spectrogram_welch(
        sig.transpose("time", "channel").values, sig.fs, **kwargs
    )
    return _wrap_3d_spectrogram(freqs, spg_time, spg, sig)


def single_spectrogram_welch(sig, **kwargs):
    """Compute a single (e.g. single-channel, single-region) spectrogram.

    Parameters
    ----------
    sig: xr.DataArray (time,) or (time, channel) where `channel` has length 1.
        Required attr: (fs, the sampling rate)
    **kwargs: optional
        Keyword arguments passed to `_compute_spectrogram_welch`.

    Returns:
    --------
    spg : xr.DataArray (frequency, time) or (frequency, time, channel)
        Spectrogram of `sig`.
    """
    if sig.values.ndim == 1:
        wrapper = _wrap_2d_spectrogram
    if sig.values.ndim == 2:
        assert (
            sig.values.shape[1] == 1
        ), "This function is not intended for multichannel data."
        wrapper = _wrap_3d_spectrogram

    freqs, spg_time, spg = tfr.compute_spectrogram_welch(
        sig.values.squeeze(), sig.fs, **kwargs
    )
    return wrapper(freqs, spg_time, spg, sig)


def get_bandpower(spg, f_range):
    """Get band-limited power from a spectrogram.

    Parameters
    ----------
    spg: xr.DataArray (frequency, time, [channel])
        Spectrogram data.
    f_range: (float, float)
        Frequency range to restrict to, as [f_low, f_high].

    Returns:
    --------
    bandpower: xr.DataArray (time, [channel])
        Sum of the power in `f_range` at each point in time.
    """
    bandpower = spg.sel(frequency=slice(*f_range)).sum(dim="frequency")
    bandpower.attrs["f_range"] = f_range

    return bandpower


def get_bandpowers(spg, bands):
    """Get multiple bandpower series in a single Dataset object.

    Examples
    --------
        get_bandpowers(spg, {'delta': (0.5, 4), 'theta': (5, 10)})
    """
    return xr.Dataset(
        {band_name: get_bandpower(spg, f_range) for band_name, f_range in bands.items()}
    )


def get_psd(spg):
    return spg.median(dim="datetime")
