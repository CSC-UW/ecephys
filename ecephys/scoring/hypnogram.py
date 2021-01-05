import numpy as np
import pandas as pd

from ..data.paths import get_datapath


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


def load_visbrain_hypnogram(path=None, subject=None, condition=None):
    """Load a Visbrain-formatted hypngoram.

    Parameters
    ----------
    path: pathlib.Path
        The path to the hypnogram.

    Returns:
    --------
    H: DataFrame
        The loaded hypnogram.
    """
    if not path:
        path = get_datapath(subject=subject, condition=condition, data="hypnogram.txt")

    H = pd.read_csv(path, sep="\t", names=["state", "end_time"], skiprows=1)
    H["start_time"] = H.apply(lambda row: get_start_time(H, row), axis=1)
    H["duration"] = H.apply(lambda row: row.end_time - row.start_time, axis=1)

    return H


def write_visbrain_hypnogram(H, path):
    H.to_csv(path, columns=["state", "end_time"], sep="\t", index=False)


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


def make_empty_hypnogram(end_time):
    """Return an empty, unscored hypnogram.

    Parameters
    ----------
    end_time: float
        The time at which the hypnogram should end, in seconds.

    Returns:
        H: pd.DataFrame
            A hypnogram containing a single state ("None") extending from t=0 until `end_time`.
    """
    H = pd.DataFrame(
        {
            "state": "None",
            "start_time": [0.0],
            "end_time": [end_time],
            "duration": [end_time],
        }
    )

    return H


def get_separated_wake_hypnogram(qwk_intervals, awk_intervals):
    """Turn a list of quiet wake and active wake intervals into a hypnogram.

    Parameters
    ----------
    qwk_intervals: list(tuple(float))
        Start and end times of each quiet wake bout.
    awk_intervals: list(tuple(float))
        Start and end times of each quiet wake bout.

    Returns
    -------
    hypnogram: pandas.DataFrame
    """

    qwk_intervals = np.asarray(qwk_intervals)
    awk_intervals = np.asarray(awk_intervals)

    qwk = pd.DataFrame(
        {
            "state": "qWk",
            "start_time": qwk_intervals[:, 0],
            "end_time": qwk_intervals[:, 1],
            "duration": np.diff(qwk_intervals).flatten(),
        }
    )
    awk = pd.DataFrame(
        {
            "state": "aWk",
            "start_time": awk_intervals[:, 0],
            "end_time": awk_intervals[:, 1],
            "duration": np.diff(awk_intervals).flatten(),
        }
    )

    return pd.concat([qwk, awk]).sort_values(by=["start_time"]).reset_index()
