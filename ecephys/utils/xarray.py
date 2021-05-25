import numpy as np
import pandas as pd
from scipy.stats import mode
from ripple_detection.core import gaussian_smooth


def estimate_fs(da):
    sample_period = mode(np.diff(da.datetime.values)).mode[0]
    assert isinstance(sample_period, np.timedelta64)
    sample_period = sample_period / pd.to_timedelta(1, "s")
    return 1 / sample_period


def get_smoothed_da(da, smoothing_sigma=10, in_place=False):
    if not in_place:
        da = da.copy()
    da.values = gaussian_smooth(da, smoothing_sigma, estimate_fs(da))
    return da


def get_smoothed_ds(ds, smoothing_sigma=10, in_place=False):
    if not in_place:
        ds = ds.copy()
    for da_name, da in ds.items():
        ds[da_name] = get_smoothed_da(da, smoothing_sigma, in_place)
    return ds