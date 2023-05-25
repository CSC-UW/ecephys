import json
import numpy as np
import xarray as xr
from pathlib import Path


def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False


def drop_unserializeable(d):
    if is_jsonable(d):
        return d
    elif isinstance(d, dict):
        return {k: drop_unserializeable(v) for k, v in d.items()}
    else:
        print(f"Dropping unserializable object: {d}")
        return ""


# TODO: Move to ecephys.one
def write_da_as_npy(da, object, dir=None):
    dir = Path().cwd() if dir is None else Path(dir)
    np.save(dir / f"{object}.data.npy", da.to_numpy())
    for coord in da.coords:
        np.save(dir / f"{object}.{coord}.npy", da[coord].to_numpy())
    metadata = {
        "name": da.name,
        "dims": [d for d in da.dims],
        "coords": [c for c in da.coords],
        "cdims": [list(v.coords.dims) for v in da.coords.values()],
        "attrs": drop_unserializeable(da.attrs),
    }
    with open(dir / f"{object}.metadata.json", "w") as f:
        json.dump(metadata, f)


# TODO: Move to ecephys.one
def read_npy_as_da(object, dir=None):
    dir = Path().cwd() if dir is None else Path(dir)
    metafile = dir / f"{object}.metadata.json"
    assert metafile.is_file(), f"Missing metadata: {metafile.name}"
    with open(metafile, "r") as f:
        meta = json.load(f)

    datafile = dir / f"{object}.data.npy"
    assert datafile.is_file(), f"Missing data: {datafile.name}"
    data = np.load(datafile)

    coords = dict()
    for coord, cdims in zip(meta["coords"], meta["cdims"]):
        coordfile = dir / f"{object}.{coord}.npy"
        assert coordfile.is_file(), f"Missing coord file: {coordfile.name}"
        coords.update({coord: (cdims, np.load(coordfile))})

    return xr.DataArray(
        data, dims=meta["dims"], coords=coords, attrs=meta["attrs"], name=meta["name"]
    )


def save_xarray_to_netcdf(xr_obj, path, **kwargs):
    """Save an Xarray object to NetCDF, which preserves obj.attrs as long as they are serializable"""
    if not (isinstance(xr_obj, xr.DataArray) or isinstance(xr_obj, xr.Dataset)):
        raise ValueError(f"Expected DataArray or Dataset, got {type(xr_obj)}.")
    Path(path).parent.mkdir(
        parents=True, exist_ok=True
    )  # Create parent directories if needed.
    xr_obj.attrs = drop_unserializeable(xr_obj.attrs)
    if isinstance(xr_obj, xr.Dataset):
        for var in xr_obj.variables:
            xr_obj[var].attrs = drop_unserializeable(xr_obj[var].attrs)
    xr_obj.to_netcdf(path, **kwargs)
    xr_obj.close()
