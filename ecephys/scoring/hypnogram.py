import numpy as np
import pandas as pd


def get_start_time(H, row):
    """Infer a bout's start time from the previous bout's end time.

    Parameters
    ----------
    H: DataFrame, (n_bouts, ?)
        Hypogram in Visbrain format with 'start_time'.
    row: Series
        A row from `H`, representing the bout that you want the start time of.

    Returns
    -------
    start_time: float
        The start time of the bout from `row`.
    """
    if row.name == 0:
        start_time = 0.0
    else:
        start_time = H.loc[row.name - 1].end_time

    return start_time


def load_visbrain_hypnogram(hypno_path):
    """Load a Visbrain-formatted hypngoram.

    Parameters
    ----------
    hypno_path: pathlib.Path
        The path to the hypnogram.

    Returns:
    --------
    H: DataFrame
        The loaded hypnogram.
    """
    H = pd.read_csv(hypno_path, sep="\t", names=["state", "end_time"], skiprows=1)
    H["start_time"] = H.apply(lambda row: get_start_time(H, row), axis=1)

    return H


def filter_states(H, states):
    """Return only hypnogram entries corresponding to desired states.

    Parameters
    ----------
    H: DataFrame, (n_bouts, ?)
        Hypnogram in Visbrain format.
    states: list of str
        The states to retain, as marked `H`.

    Returns
    -------
    H: DataFrame, (<= n_bouts, ?)
        The hypnogram, with only desired states retained.
    """

    return H[H["state"].isin(states)]


def mask_times(H, states, times):
    """Return a mask that is true where times belong to specific states.

    Parameters
    ----------
    H: DataFrame, (n_bouts, ?)
        Hypnogram in Visbrain format with 'start_time' and 'end_time' fields.
    states: list of str
        The states to mask, as marked `H`.
    times: (n_samples,)
        The times to mask.

    Returns
    -------
    mask: (n_samples,)
        True where `times` belong to one of the indicated states, false otherise.
    """
    mask = np.full_like(times, False, dtype=bool)
    for bout in filter_states(H, states).itertuples():
        mask[np.logical_and(times >= bout.start_time, times <= bout.end_time)] = True

    return mask