import numpy as np
import pandas as pd
from pathlib import Path


class Hypnogram(pd.DataFrame):
    def __init__(self, *args, **kwargs):
        super(Hypnogram, self).__init__(*args, **kwargs)

    def _validate(self):
        if not {"state", "start_time", "end_time", "duration"}.issubset(self):
            raise AttributeError()

    @property
    def _constructor(self):
        return Hypnogram

    def filter_by_state(self, states):
        """Return all bouts of the given states.

        Parameters:
        -----------
        states: list of str
        """
        return self[self["state"].isin(states)]

    def mask_times_by_state(self, times, states):
        """Return a mask that is true where times belong to specific states.

        Parameters
        ----------
        times: (n_times,)
            The times to mask.
        states: list of str
            The states of interest.

        Returns
        -------
        (n_times,)
            True where `times` belong to one of the indicated states, false otherise.
        """
        mask = np.full_like(times, False, dtype=bool)
        for bout in self.filter_by_state(states).itertuples():
            mask[(times >= bout.start_time) & (times <= bout.end_time)] = True

        return mask

    def get_states(self, times):
        """Given an array of times, label each time with its state.

        Parameters:
        -----------
        times: (n_times,)
            The times to label.

        Returns:
        --------
        states (n_times,)
            The state label for each sample in `times`.
        """
        labels = pd.Series([""] * len(times))
        for bout in self.itertuples():
            times_in_bout = (times >= bout.start_time) & (times <= bout.end_time)
            labels.values[times_in_bout] = bout.state

        return labels

    def write(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.to_csv(
            path,
            sep="\t",
            index=False,
        )


class FloatHypnogram(Hypnogram):
    @property
    def _constructor(self):
        return FloatHypnogram

    def write_visbrain(self, path):
        path(path).parent.mkdir(parents=True, exist_ok=True)
        self.to_csv(path, columns=["state", "end_time"], sep="\t", index=False)

    def as_datetime(self, start_datetime):
        df = self.copy()
        df["start_time"] = start_datetime + pd.to_timedelta(df["start_time"], "s")
        df["end_time"] = start_datetime + pd.to_timedelta(df["end_time"], "s")
        df["duration"] = pd.to_timedelta(df["duration"], "s")
        return DatetimeHypnogram(df)


class DatetimeHypnogram(Hypnogram):
    @property
    def _constructor(self):
        return DatetimeHypnogram

    def keep_first(self, states, cumulative_duration):
        """Keep bouts of given states until a cumulative duration is reached.

        Parameters:
        -----------
        states: list of str
            States which count towards the cumulative time limit.
        cumulative_duration:
            Any valid timedelta specifier, `02:30:10` for 2h, 30m, 10s.
        """
        filt = self.filter_by_state(states)
        keep = filt.duration.cumsum() <= pd.to_timedelta(cumulative_duration)
        return self.loc[keep]


def _infer_bout_start(df, bout):
    """Infer a bout's start time from the previous bout's end time.

    Parameters
    ----------
    h: DataFrame, (n_bouts, ?)
        Hypogram in Visbrain format with 'start_time'.
    row: Series
        A row from `h`, representing the bout that you want the start time of.

    Returns
    -------
    start_time: float
        The start time of the bout from `row`.
    """
    if bout.name == 0:
        start_time = 0.0
    else:
        start_time = df.loc[bout.name - 1].end_time

    return start_time


def load_visbrain_hypnogram(path):
    """Load a Visbrain formatted hypnogram."""
    df = pd.read_csv(path, sep="\t", names=["state", "end_time"], comment="*")
    df["start_time"] = df.apply(lambda row: _infer_bout_start(df, row), axis=1)
    df["duration"] = df.apply(lambda row: row.end_time - row.start_time, axis=1)
    return FloatHypnogram(df)


def load_spike2_hypnogram(path):
    """Load a Spike2 formatted hypnogram."""
    df = pd.read_table(
        path,
        sep="\t",
        names=["epoch", "start_time", "end_time", "state", "comment", "blank"],
        usecols=["epoch", "start_time", "end_time", "state"],
        index_col="epoch",
        skiprows=22,
    )
    return FloatHypnogram(df)


def load_datetime_hypnogram(path):
    """Load a hypnogram whose entries are valid datetime strings."""
    df = pd.read_csv(path, sep="\t")
    df["start_time"] = pd.to_datetime(df["start_time"])
    df["end_time"] = pd.to_datetime(df["end_time"])
    df["duration"] = pd.to_timedelta(df["duration"])
    return DatetimeHypnogram(df)


def get_empty_hypnogram(end_time):
    """Return an empty, unscored hypnogram.

    Parameters
    ----------
    end_time: float
        The time at which the hypnogram should end, in seconds.

    Returns:
        H: pd.DataFrame
            A hypnogram containing a single state ("None") extending from t=0 until `end_time`.
    """
    df = pd.DataFrame(
        {
            "state": "None",
            "start_time": [0.0],
            "end_time": [end_time],
            "duration": [end_time],
        }
    )
    return FloatHypnogram(df)


def get_separated_wake_hypnogram(qwk_intervals, awk_intervals):
    """Turn a list of quiet wake and active wake intervals into a hypnogram.

    Parameters
    ----------
    qwk_intervals: list(tuple(float))
        Start and end times of each quiet wake bout.
    awk_intervals: list(tuple(float))
        Start and end times of each quiet wake bout.
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

    df = pd.concat([qwk, awk]).sort_values(by=["start_time"]).reset_index()
    return Hypnogram(df)
