import pandas as pd
import xarray as xr
import numpy as np
from ecephys import hypnogram as hg


#####
# DataArray utils
#####
# TODO: Add as methods to DataArrayWrapper?

def get_dim_coords(da, dim_name):
    """Get all the coordinates corresponding to one dimension, as a dict that can be assigned to a new xarray object using `assign_coords`."""
    if dim_name:
        return {coord_name: coord_obj for coord_name, coord_obj in da.coords.items() if dim_name in coord_obj[coord_name].dims}
    else:
        return {coord_name: coord_obj for coord_name, coord_obj in da.coords.items() if coord_obj[coord_name].dims == ()}

def get_boundary_ilocs(da, coord_name):
    df = da[coord_name].to_dataframe() # Just so we can use pandas utils
    df = df.loc[:,~df.columns.duplicated()].copy() # Drop duplicate columns
    df = df.reset_index() # So we can find boundaries of dimensions too
    changed = df[coord_name].ne(df[coord_name].shift().bfill())
    boundary_locs = df[coord_name][changed.shift(-1, fill_value=True)].index
    return np.where(np.isin(df.index, boundary_locs))[0]

#####
# Dataset utils
#####

def load_and_concatenate_datasets(paths):
    datasets = list()
    for path in paths:
        try:
            datasets.append(xr.load_dataset(path))
        except FileNotFoundError:
            pass

    return rebase_time(xr.concat(datasets, dim="time"))

#####
# Voltage timeseries utils
#####
# TODO: Add as a method to LFP class

def rereference(sig, ref_chans, func=np.mean):
    """Re-reference a signal to a function of specific channels."""
    # Example: For common avg. ref., ref_chans == sig.channel
    ref = sig.sel(channel=ref_chans).reduce(func, dim="channel", keepdims=True)
    return sig - ref.values


#####
# Hypnogram utils
#####
# TODO: Add as methods to a Datetimed class

# These functions could use the XRSig accessor instead of requiring
# datetime to be a dimension.

# These functions could also use use index masking instead of requiring
# datetime to be a dimension. For example: dat.isel(time=keep).


def add_states(dat, hypnogram):
    """Annotate each timepoint in the dataset with the corresponding state label.

    Parameters:
    -----------
    dat: Dataset or DataArray with dimension `datetime`.
    hypnogram: DatetimeHypnogram

    Returns:
    --------
    xarray object with new coordinate `state` on dimension `datetime`.
    """
    assert isinstance(hypnogram, hg.DatetimeHypnogram)
    assert "datetime" in dat.dims, "Data must contain datetime dimension."
    states = hypnogram.get_states(dat.datetime)
    return dat.assign_coords(state=("datetime", states))


def keep_states(dat, hypnogram, states):
    """Select only timepoints corresponding to desired states.

    Parameters:
    -----------
    dat: Dataset or DataArray with dimension `datetime`
    hypnogram: DatetimeHypnogram
    states: list of strings
        The states to retain.
    """
    assert isinstance(hypnogram, hg.DatetimeHypnogram)
    assert "datetime" in dat.dims, "Data must contain datetime dimension."
    keep = hypnogram.keep_states(states).covers_time(dat.datetime)
    return dat.sel(datetime=keep)


def keep_hypnogram_contents(dat, hypnogram):
    """Select only timepoints covered by the hypnogram.

    Parameters:
    -----------
    dat: Dataset or DataArray with dimension `datetime`
    hypnogram: DatetimeHypnogram
    """
    assert isinstance(hypnogram, hg.DatetimeHypnogram)
    assert "datetime" in dat.dims, "Data must contain datetime dimension."
    keep = hypnogram.covers_time(dat.datetime)
    return dat.sel(datetime=keep)

#####
# Likely deprecated utils
#####

# Is this needed anymore, now that it exists in sglxarray?
def rebase_time(sig, in_place=True):
    """Rebase time and timedelta coordinates so that t=0 corresponds to the beginning
    of the datetime dimension."""
    if not in_place:
        sig = sig.copy()
    sig["timedelta"] = sig.datetime - sig.datetime.min()
    sig["time"] = sig["timedelta"] / pd.to_timedelta(1, "s")
    return sig


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