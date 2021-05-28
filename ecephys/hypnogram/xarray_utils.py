import numpy as np


def add_states_to_dataset(ds, hypnogram):
    """Annotate each timepoint in the dataset with the corresponding state label.

    Parameters:
    -----------
    ds: xr.Dataset with dimension `time`.
        The data to annotate.
    hypnogram: DataFrame, (n_bouts, ?)
        Hypnogram in Visbrain format with `start_time` and `end_time` formats that
        match `ds`.

    Returns:
    --------
    ds: xr.Dataset with new coordinate `state`.
    """
    states = hypnogram.get_states(ds.time)
    return ds.assign_coords(state=("time", states))


def filter_dataset_by_state(ds, hypnogram, states):
    """Select only timepoints corresponding to desired states.

    Parameters:
    -----------
    ds: xr.Dataset
    hypnogram: pd.DataFrame
    states: list of strings
        The states to retain.

    Returns:
    --------
    ds: xr.Dataset
        A dataset with the same dimensions as `ds`, with values that do not
        correspond to `states` set to nan.
    """
    labels = hypnogram.get_states(ds.time)
    return ds.where(np.isin(labels, states))
