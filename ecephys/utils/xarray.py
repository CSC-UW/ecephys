from pathlib import Path

import xarray as xr

from . import misc


def save_xarray_to_netcdf(xr_obj, path, **kwargs):
    """Save an Xarray object to NetCDF, which preserves obj.attrs as long as they are serializable"""
    if not (isinstance(xr_obj, xr.DataArray) or isinstance(xr_obj, xr.Dataset)):
        raise ValueError(f"Expected DataArray or Dataset, got {type(xr_obj)}.")
    Path(path).parent.mkdir(
        parents=True, exist_ok=True
    )  # Create parent directories if needed.
    xr_obj.attrs = misc.drop_unjsonable(xr_obj.attrs)
    if isinstance(xr_obj, xr.Dataset):
        for var in xr_obj.variables:
            xr_obj[var].attrs = misc.drop_unjsonable(xr_obj[var].attrs)
    xr_obj.to_netcdf(path, **kwargs)
    xr_obj.close()
