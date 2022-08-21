import xarray as xr
import pandas as pd
import numpy as np
from ..signal import timefrequency as tfr
from . import core

#####
# Likely deprecated utils
#####

# TODO: Remove this function. It is ugly and dangerous. Function graveyard gist?
def dtdim(dat):
    assert "datetime" in dat.coords, "Datetime coordinate not found."
    if "time" in dat.dims:
        return dat.swap_dims({"time": "datetime"})
    elif "timedelta" in dat.dims:
        return dat.swap_dims({"timedelta": "datetime"})
    else:
        raise ValueError(
            "Exactly one of `time` or `timedelta` must be present as a dimension."
        )


# TODO: This is not necessary, since you can achieve the same with:
# da2 = xr.DataArray(coords={**da1.foo.coords}), or da2['foo'] = da1['foo']
def get_dim_coords(da, dim_name):
    """Get all the coordinates corresponding to one dimension, as a dict that can be assigned to a new xarray object using `assign_coords`."""
    if dim_name:
        return {
            coord_name: coord_obj
            for coord_name, coord_obj in da.coords.items()
            if dim_name in coord_obj[coord_name].dims
        }
    else:
        return {
            coord_name: coord_obj
            for coord_name, coord_obj in da.coords.items()
            if coord_obj[coord_name].dims == ()
        }


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

    freqs, spg_time, spg = tfr.single_spectrogram_welch(
        sig.values.squeeze(), sig.fs, **kwargs
    )
    return wrapper(freqs, spg_time, spg, sig)


class SPGs(core.TimeseriesND):
    """Multsignal spectrograms.
    Dimensions: ('frequency', 'time', <signal>).
    Attributes:
        fs: Sampling rate

    'time' is seconds from a reference period (usually the start of the experiment).
    """

    def __init__(self, da):
        super().__init__(da)
        self._validate()

    def _validate(self):
        expected = ("frequency", "time")
        if not self.dims[:2] == expected:
            raise AttributeError(
                f"{self.__class__.__name__} must have first dimensions {expected}."
            )

    def spectra(self):
        spectra = self.median(dim="time")
        return PSDs(spectra)

    def bandpowers(self, fRange):
        """Get band-limited power from a spectrogram.

        Parameters
        ----------
        f_range: (float, float)
            Frequency range to restrict to, as [f_low, f_high].

        Returns:
        --------
        bandpower: xr.DataArray (time, channel)
            Sum of the power in `f_range` at each point in time.
        """
        bandpower = self.sel(frequency=slice(*fRange)).sum(dim="frequency")
        bandpower.attrs["fRange"] = fRange
        return bandpower

    def multiband_powers(self, bands):
        """Get multiple bandpower series in a single Dataset object.

        Examples
        --------
            self.multiband_powers({'delta': (0.5, 4), 'theta': (5, 10)})
        """
        return xr.Dataset(
            {
                band_name: self.bandpowers(f_range)
                for band_name, f_range in bands.items()
            }
        )


class PSDs(core.DataArrayWrapper):
    """Channelwise power spectral densities.
    Dimensions: ('frequency', <signal>)."""

    def __init__(self, da):
        super().__init__(da)
        self._validate()

    def _validate(self):
        expected = ("frequency", "channel")
        if not self.dims == expected:
            raise AttributeError(
                f"{self.__class__.__name__} must have dimensions {expected}."
            )

    def bandpower(self, slice):
        pow = self.sel(frequency=slice).sum(dim="frequency")
        return core.LaminarScalars(pow)


class Spectrogram(core.DataArrayWrapper):
    """Single spectrogram, for a channel or region.
    Dimensions: ('frequency', 'time').
    'time' is seconds from a reference period (usually the start of the experiment)."""

    def __init__(self, da):
        super().__init__(da)
        self._validate()

    def _validate(self):
        expected = ("frequency", "time")
        if not self.dims == expected:
            raise AttributeError(
                f"{self.__class__.__name__} must have dimensions {expected}.\n"
                f"Got dimensions {self.dims}."
            )

    def bandpower(self, f_range):
        pow = self.sel(frequency=slice(*f_range)).sum(dim="frequency")
        return pow
