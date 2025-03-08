import dask.array
import numpy as np
import xarray as xr
import zarr


def _get_increasing_segments_mask(times):
    keep = np.ones((len(times),), dtype=bool)
    gap_ixs = np.where(np.diff(times) < 0)[0]
    for gap_ix in gap_ixs:
        pre_gap_values = times[gap_ix]
        next_ix = np.where(times[gap_ix:] > pre_gap_values)[0][0]
        keep[gap_ix + 1 : gap_ix + next_ix] = False
    return keep


def _hotfix_times(da):
    keep = _get_increasing_segments_mask(da.time.data)
    return da.sel(time=keep).copy()


def open_processed_zarr_as_xarray(fpath):
    """Load SI-saved zarr as dask-based xarray for OFF detection.

    NB: Non monotonously increasing timestamps are dismissed."""

    assert fpath.exists()
    zg = zarr.open(fpath)
    da = xr.DataArray(
        data=dask.array.from_zarr(zg.traces_seg0).rechunk(),
        dims=("time", "channel"),
        coords={
            "time": zg.times_seg0,
            "channel": zg.properties["channel_name"],
            "y": ("channel", zg.properties["location"][:, 1]),
        },
        attrs={
            "units": "AU",
            "fs": zg.attrs["sampling_frequency"],
        },
        name="Processed AP",
    )
    da = _hotfix_times(da)

    return da
