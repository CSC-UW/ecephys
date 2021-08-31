import xarray as xr
import pandas as pd
from ..signal import timefrequency as tfr


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
    time = sig.time.values.min() + spg_time
    timedelta = sig.timedelta.values.min() + pd.to_timedelta(spg_time, "s")
    datetime = sig.datetime.values.min() + pd.to_timedelta(spg_time, "s")
    return xr.DataArray(
        spg,
        dims=("frequency", "time", "channel"),
        coords={
            "frequency": freqs,
            "time": time,
            "channel": sig.channel.values,
            "timedelta": ("time", timedelta),
            "datetime": ("time", datetime),
        },
        attrs={"units": f"{sig.units}^2/Hz"},
    )


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
