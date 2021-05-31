import pandas as pd
import numpy as np
from hypnogram import DatetimeHypnogram


def rebase_time(sig, in_place=True):
    if not in_place:
        sig = sig.copy()
    sig["timedelta"] = sig.datetime - sig.datetime.min()
    sig["time"] = sig["timedelta"] / pd.to_timedelta(1, "s")
    return sig


def add_states_to_dataset(ds, hypnogram):
    """Annotate each timepoint in the dataset with the corresponding state label.

    Parameters:
    -----------
    ds: xr.Dataset with coordinate `datetime` on dimension `time`.
        The data to annotate.
    hypnogram: DatetimeHypnogram
        Hypnogram with states to add.

    Returns:
    --------
    ds: xr.Dataset with new coordinate `state` on dimension `time`.
    """
    assert isinstance(hypnogram, DatetimeHypnogram)
    states = hypnogram.get_states(ds.datetime)
    return ds.assign_coords(state=("time", states))


def filter_dataset_by_state(ds, hypnogram, states):
    """Select only timepoints corresponding to desired states.

    Parameters:
    -----------
    ds: xr.Dataset with dimension `datetime`
        The data to filder
    hypnogram: DatetimeHypnogram
    states: list of strings
        The states to retain.

    Returns:
    --------
    ds: xr.Dataset
        A dataset with the same dimensions as `ds`, and all values that do not
        correspond to `states` dropped.
    """
    assert isinstance(hypnogram, DatetimeHypnogram)
    labels = hypnogram.get_states(ds.datetime)
    return ds.where(np.isin(labels, states)).dropna(dim="time")


def filter_dataset_by_hypnogram(ds, hypnogram):
    """Select only timepoints covered by the hypnogram.

    Parameters:
    -----------
    ds: xr.Dataset with dimension `datetime`
        The data to filder
    hypnogram: DatetimeHypnogram

    Returns:
    --------
    ds: xr.Dataset
        A dataset with the same dimensions as `ds`, and all values that do not
        correspond to bouts in the hypnogram dropped.
    """
    assert isinstance(hypnogram, DatetimeHypnogram)
    keep = np.full_like(ds.datetime, False)
    for bout in hypnogram.itertuples():
        times_in_bout = (ds.datetime >= bout.start_time) * (
            ds.datetime <= bout.end_time
        )
        keep[times_in_bout] = True

    return ds.where(keep).dropna(dim="time")