import numpy as np
from .hypnogram import DatetimeHypnogram


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
        A dataset with the same dimensions as `ds`, with values that do not
        correspond to `states` set to nan.
    """
    assert isinstance(hypnogram, DatetimeHypnogram)
    labels = hypnogram.get_states(ds.datetime)
    return ds.where(np.isin(labels, states)).dropna(dim="time")
