import numpy as np
import pandas as pd
import xarray as xr
from emg_from_lfp import compute_emg


def get_emg(sig, **emg_kwargs):
    """Estimate the EMG from LFP signals, using the `emg_from_lfp` package.
    Paramters used for the computation are stored as attrs in the returned
    DataArray.

    Parameters:
    -----------
    sig: DataArray
        Must has sampling rate `fs`, dimensions `time` and `channel`.
    **emg_kwargs:
        Keyword arguments passed to `emg_from_lfp.compute_emg()`

    Returns:
    --------
    DataArray:
        EMG with time dimensiona and timedelta, datetime coords.
    """
    values = compute_emg(
        sig.transpose("channel", "time").values, sig.fs, **emg_kwargs
    ).flatten()
    time = np.linspace(sig.time.min(), sig.time.max(), values.size)
    timedelta = pd.to_timedelta(time, "s")
    datetime = sig.datetime.values.min() + timedelta

    emg = xr.DataArray(
        values,
        dims="time",
        coords={
            "time": time,
            "timedelta": ("time", timedelta),
            "datetime": ("time", datetime),
        },
        attrs={"long_name": "emg_from_lfp", "units": "zero-lag correlation"},
    )
    for key in emg_kwargs:
        emg.attrs[key] = emg_kwargs[key]

    return emg