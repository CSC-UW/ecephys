from __future__ import annotations
import numpy as np
import pandas as pd
import datetime
import warnings
from ecephys import utils
from pathlib import Path
from pandas.core.dtypes.common import (
    is_datetime64_any_dtype,
)


class Hypnogram:
    def __init__(self, df):
        self._df = df
        self._validate()

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self._df, attr)

    def __getitem__(self, item):
        return self._df[item]

    def __setitem__(self, item, data):
        self._df[item] = data

    def __repr__(self):
        return repr(self._df)

    def __len__(self):
        return len(self._df)

    def _validate(self):
        if not {"state", "start_time", "end_time", "duration"}.issubset(self._df):
            raise AttributeError(
                "Required columns `state`, `start_time`, `end_time`, and `duration` are not present."
            )
        if not all(self._df["start_time"] <= self._df["end_time"]):
            raise ValueError("Not all start times precede end times.")
        if not self._df["start_time"].is_monotonic_increasing:
            raise ValueError("Hypnogram start times are not monotonically increasing.")
        if not self._df["end_time"].is_monotonic_increasing:
            raise ValueError("Hypnogram end times are not monotonically increasing.")

    def write_htsv(self, file):
        """Write as HTSV."""
        file = Path(file)
        assert file.suffix == ".htsv", "File must use extension .htsv"
        file.parent.mkdir(parents=True, exist_ok=True)
        self.to_csv(
            file,
            sep="\t",
            header=True,
            index=False,
        )

    def keep_states(self, states):
        """Return all bouts of the given states.
        Parameters:
        -----------
        states: list of str
        """
        return self.__class__(self._df[self._df["state"].isin(states)])

    def drop_states(self, states):
        """Drop all bouts of the given states.
        Parameters:
        -----------
        states: list of str
        """
        return self.__class__(self._df[~self._df["state"].isin(states)])

    def replace_states(self, replacement_dict):
        """Takes a dict where keys are current states and values are desired states, and updates the hypnogram accoridngly."""
        return self.__class__(self._df.replace(replacement_dict))

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
        for bout in self.keep_states(states).itertuples():
            mask[(times >= bout.start_time) & (times <= bout.end_time)] = True

        return mask

    def get_states(self, times: np.ndarray, default_value="") -> np.ndarray:
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
        assert np.all(np.diff(times) >= 0), "The times must be increasing."
        assert times.ndim == 1

        epoch_idxs = np.searchsorted(
            times, np.c_[self.start_time.to_numpy(), self.end_time.to_numpy()]
        )
        dtype = "object" if isinstance(default_value, str) else None
        result = np.full_like(times, fill_value=default_value, dtype=dtype)
        for i, ep in enumerate(epoch_idxs):
            result[ep[0] : ep[1]] = self.state.iloc[i]
        return result

    def covers_time(self, times):
        """Given an array of times, return True where that time is covered by
        the hypnogram."""
        covered = np.full_like(times, False, dtype="bool")
        for bout in self.itertuples():
            times_in_bout = (times >= bout.start_time) & (times <= bout.end_time)
            covered[times_in_bout] = True

        return covered

    def fractional_occupancy(self, ignore_gaps=True):
        """Return a DataFrame with the time spent in each state, as a fraction of
        the total time covered by the hypnogram.
        Parameters:
        -----------
        ignore_gaps: bool
            If True, unscored gaps do not contribute to total time.
        """
        total_time = (
            self.duration.sum()
            if ignore_gaps
            else self.end_time.max() - self.start_time.min()
        )
        return self.groupby("state").duration.sum() / total_time

    def reconcile(self, other, how="self"):
        """Reconcile this hypnogram with another, per `reconcile_hypnograms`.

        Parameters:
        -----------
        other: Hypnogram
        how: str ('sel' or 'other')
            If 'self', resolve any conflicts in favor of this hypnogram.
            If 'other', resolve any conflicts in favor of `other`.
            Default: 'self'

        Returns:
        --------
        Hypnogram
        """
        assert type(self) == type(
            other
        ), "Cannot reconcile hypnograms of different types."
        if how == "self":
            return self.__class__(reconcile_hypnograms(self, other))
        elif how == "other":
            return self.__class__(reconcile_hypnograms(other, self))
        else:
            raise ValueError(
                f"Argument `how` should be either 'sel' or 'other'. Got {how}."
            )

    def get_consolidated(
        self,
        states,
        minimum_time,
        minimum_endpoint_bout_duration,
        maximum_antistate_bout_duration,
        frac=0.8,
    ):
        """Get periods of consolidated sleep, wake, or any arbitrary set of states.

        A period is considered consolidated if more than a given fraction of its duration
        (e.g. frac=0.8 or 80%) is spent in the state(s) of interest, and the cumulative
        amount of time spent in the state(s) of interest exceeds `minimum_time`.
        Additionally, a consolidated period must be maximal, i.e. it cannot be contained by
        a longer consolidated period.

        Parameters:
        -----------
        states: list of str
            The states of interest.
        minimum_time:
            The minimum cumulative time that must be spent in the states of interest for
            a given period to be considered consolidated.
        minimum_endpoint_bout_duration:
            The minimum length of any bout that can be used as the start or end of a consolidated period.
            This prevents picking bouts from fragmented periods (e.g. falling asleep after a period of extended wake), and also drastically reduces computaiton time.
        maximum_antistate_bout_duration:
            Do not allow periods to contain any bouts of unwanted states longer
            than a given duration.
        frac: float between 0 and 1
            The minimum fraction of a given period that must be spent in the states of
            interest for that period to be considered consolidated.

        Returns:
        --------
        matches: list of pd.DataFrame
            Each DataFrame is a slice of the hypnogram, corresponding to a consolidated
            period.
        """
        zero = np.array([0], dtype=self.duration.dtype)[
            0
        ]  # Represents duration of length 0, regardless of dtype
        assert (
            self.start_time.is_monotonic_increasing
        ), "Hypnogram must be sorted by start_time."
        endpoint_bouts = self.keep_states(states).keep_longer(
            minimum_endpoint_bout_duration
        )
        k = endpoint_bouts.index.min() - 1
        matches = list()
        # i = period start, j = period end, k = end of last consolidated period
        for i in endpoint_bouts.index:
            if i <= k:
                continue
            for j in endpoint_bouts.index[::-1]:
                if j < np.max([i, k]):
                    break
                isostate_bouts = self.__class__(self.loc[i:j]).keep_states(states)
                time_in_states = np.max([isostate_bouts.duration.sum(), zero])
                if time_in_states < minimum_time:
                    break  # because all other periods in the loop will also fail
                antistate_bouts = self.__class__(self.loc[i:j]).drop_states(states)
                if antistate_bouts.duration.max() > maximum_antistate_bout_duration:
                    continue
                total_time = (
                    self.loc[i:j].end_time.max() - self.loc[i:j].start_time.min()
                )
                if (time_in_states / total_time) >= frac:
                    matches.append(self.__class__(self.loc[i:j]))
                    k = j
                    break  # don't bother checking subperiods of good periods
        return matches


class FloatHypnogram(Hypnogram):
    def write_visbrain(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._df.to_csv(
            path, columns=["state", "end_time"], sep="\t", index=False, header=False
        )

    def as_datetime(self, start_datetime):
        df = self._df.copy()
        df["start_time"] = start_datetime + pd.to_timedelta(df["start_time"], "s")
        df["end_time"] = start_datetime + pd.to_timedelta(df["end_time"], "s")
        df["duration"] = pd.to_timedelta(df["duration"], "s")
        return DatetimeHypnogram(df)

    def keep_longer(self, duration):
        """Keep bouts longer than a given duration.

        Parameters:
        -----------
        duration: float
        """
        return self.__class__(self.loc[self.duration > duration])

    def keep_between_time(self, start_time=None, end_time=None):
        """Keep all hypnogram bouts that fall between two times

        Paramters:
        ----------
        start_time: in seconds
        end_time: in seconds
        """
        if start_time is None:
            start_time = self.start_time.min()
        if end_time is None:
            end_time = self.end_time.max()

        keep = (self.start_time >= start_time) & (self.end_time <= end_time)
        return self.__class__(self._df[keep])

    def get_consolidated(
        self,
        states,
        minimum_time=0,
        minimum_endpoint_bout_duration=0,
        maximum_antistate_bout_duration=np.inf,
        frac=0.8,
    ):
        """See Hypnogram.get_consolidated"""
        return Hypnogram.get_consolidated(
            self,
            states,
            minimum_time,
            minimum_endpoint_bout_duration,
            maximum_antistate_bout_duration,
            frac=frac,
        )

    @classmethod
    def clean(cls, df: pd.DataFrame) -> FloatHypnogram:
        return cls(clean(df, condenseTol=1, missingDataTol=1, zero=0))

    @classmethod
    def get_dummy(cls, start_time=0.0, end_time=np.Inf):
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
                "start_time": [float(start_time)],
                "end_time": [float(end_time)],
                "duration": [float(end_time - start_time)],
            }
        )
        return cls(df)

    @classmethod
    def from_htsv(cls, file):
        assert Path(file).suffix == ".htsv", "File must use extension .htsv"
        df = pd.read_csv(file, sep="\t", header=0)
        return cls(df)

    @classmethod
    def from_visbrain(cls, file):
        """Load a Visbrain formatted hypnogram."""
        df = pd.read_csv(file, sep="\t", names=["state", "end_time"], comment="*")

        def _infer_start(df: pd.DataFrame, bout: pd.Series) -> float:
            return 0.0 if bout.name == 0 else df.loc[bout.name - 1].end_time

        df["start_time"] = df.apply(lambda row: _infer_start(df, row), axis=1)
        df["duration"] = df.apply(lambda row: row.end_time - row.start_time, axis=1)
        return cls(df)

    @classmethod
    def from_Spike2(cls, file):
        """Load a Spike2 formatted hypnogram."""
        df = pd.read_table(
            file,
            sep="\t",
            names=["epoch", "start_time", "end_time", "state", "comment", "blank"],
            usecols=["epoch", "start_time", "end_time", "state"],
            index_col="epoch",
            skiprows=22,
        )
        return cls(df)

    @classmethod
    def from_SleepSign(cls, file):
        """Load a SleepSign hypnogram, exported using the `trend` function."""
        df = pd.read_table(
            file,
            skiprows=19,
            usecols=[0, 1, 2],
            names=["start_time", "epoch", "state"],
            parse_dates=["start_time"],
            index_col="epoch",
        )
        # Make sure that the data starts with epoch 0.
        assert (
            df.index.values[0] == 0
        ), "First epoch found is not #0. Unexpected number of header lines in file?"

        # The datetimes in the first column are meaningless. Convert them to floats.
        df["start_time"] = (df.start_time - df.start_time[0]) / pd.to_timedelta(1, "s")

        # Make sure all epochs are the same length, so that we can safely infer the file's end time.
        def _all_equal(iterator):
            """Check if all items in an un-nested array are equal."""
            try:
                iterator = iter(iterator)
                first = next(iterator)
                return all(first == rest for rest in iterator)
            except StopIteration:
                return True

        epoch_lengths = df.start_time.diff().values[1:]
        assert _all_equal(epoch_lengths), "Epochs are not all the same length."
        epoch_length = epoch_lengths[0]

        # Infer the epoch end times, and compute epoch durations
        df["end_time"] = df.start_time + epoch_length
        df["duration"] = df.apply(lambda row: row.end_time - row.start_time, axis=1)
        assert all(df.duration == epoch_length)

        # Reorder columns and return
        df = df[["state", "start_time", "end_time", "duration"]]
        return cls(df)


class DatetimeHypnogram(Hypnogram):
    def as_float(self):
        df = self._df.copy()
        start_datetime = df.start_time.min()
        df["start_time"] = (df.start_time - start_datetime) / pd.to_timedelta("1s")
        df["end_time"] = (df.end_time - start_datetime) / pd.to_timedelta("1s")
        df["duration"] = df.duration / pd.to_timedelta("1s")
        return FloatHypnogram(df)

    def keep_first(self, cumulative_duration, trim=True):
        """Keep hypnogram bouts until a cumulative duration is reached.

        Parameters:
        -----------
        cumulative_duration:
            Any valid timedelta specifier, `02:30:10` for 2h, 30m, 10s.
        """
        if trim:
            excess = self.duration.cumsum() - pd.to_timedelta(cumulative_duration)
            is_excess = excess > pd.to_timedelta(0)
            if not is_excess.any():
                return self
            amount_to_trim = excess[is_excess].min()
            trim_until = self.loc[is_excess].end_time.min() - amount_to_trim
            new = trim_hypnogram(self._df, self.start_time.min(), trim_until)
        else:
            keep = self.duration.cumsum() <= pd.to_timedelta(cumulative_duration)
            new = self.loc[keep]
        return self.__class__(new)

    def keep_last(self, cumulative_duration, trim=True):
        """Keep only a given amount of time at the end of a hypnogram.

        Parameters:
        -----------
        cumulative_duration:
            Any valid timedelta specifier, `02:30:10` for 2h, 30m, 10s.
        """
        if trim:
            excess = self.duration[::-1].cumsum() - pd.to_timedelta(cumulative_duration)
            is_excess = excess > pd.to_timedelta(0)
            if not is_excess.any():
                return self
            amount_to_trim = excess[is_excess].min()
            trim_until = self.loc[is_excess].start_time.max() + amount_to_trim
            new = trim_hypnogram(self._df, trim_until, self.end_time.max())
        else:
            keep = np.cumsum(self.duration[::-1])[::-1] <= pd.to_timedelta(
                cumulative_duration
            )
            new = self.loc[keep]
        return self.__class__(new)

    def keep_between_time(self, start_time, end_time):
        """Keep all hypnogram bouts that fall between two times of day.
        Analagous to `pandas.DataFrame.between_time`.

        Paramters:
        ----------
        start_time:
            The starting hour, e.g. '13:00:00' for 1PM.
        end_time:
            The ending hour, e.g. '14:00:00' for 2PM.
        """
        start_time, end_time = _check_time(start_time), _check_time(end_time)
        keep = np.intersect1d(
            pd.DatetimeIndex(self.start_time).indexer_between_time(
                start_time, end_time
            ),
            pd.DatetimeIndex(self.end_time).indexer_between_time(start_time, end_time),
        )
        return self.__class__(self.iloc[keep])

    def keep_between_datetime(self, start_time=None, end_time=None):
        """Keep all hypnogram bouts that fall between two datetimes.

        Paramters:
        ----------
        start_time: datetime, or str
            The starting time, either as a datetime object or as a datetime string, e.g. '2021-12-30T22:00:01'
        end_time:
            The ending time, either as a datetime object or as a datetime string, e.g. '2021-12-30T22:00:01'
        """
        if start_time is None:
            start_time = self.start_time.min()
        if end_time is None:
            end_time = self.end_time.max()
        start_time, end_time = _check_datetime(start_time), _check_datetime(end_time)
        keep = (self.start_time >= start_time) & (self.end_time <= end_time)
        return self.__class__(self.loc[keep])

    def keep_longer(self, duration):
        """Keep bouts longer than a given duration.

        Parameters:
        -----------
        duration:
            Any valid timedelta specifier, `02:30:10` for 2h, 30m, 10s.
        """
        return self.__class__(self.loc[self.duration > pd.to_timedelta(duration)])

    def get_consolidated(
        self,
        states,
        minimum_time="0S",
        minimum_endpoint_bout_duration="0S",
        maximum_antistate_bout_duration=pd.Timedelta.max,
        frac=0.8,
    ):
        return Hypnogram.get_consolidated(
            self,
            states,
            pd.to_timedelta(minimum_time),
            pd.to_timedelta(minimum_endpoint_bout_duration),
            pd.to_timedelta(maximum_antistate_bout_duration),
            frac=frac,
        )

    @classmethod
    def clean(cls, df: pd.DataFrame):
        return cls(
            clean(
                df,
                condenseTol=pd.to_timedelta("1s"),
                missingDataTol=pd.to_timedelta("1s"),
                zero=pd.to_timedelta("0s"),
            )
        )

    @classmethod
    def from_htsv(cls, file):
        """Load a hypnogram whose entries are valid datetime strings."""
        assert Path(file).suffix == ".htsv", "File must use extension .htsv"
        try:
            df = pd.read_csv(file, sep="\t", header=0)
        except pd.errors.EmptyDataError:
            return None

        df["start_time"] = pd.to_datetime(df["start_time"])
        df["end_time"] = pd.to_datetime(df["end_time"])
        df["duration"] = pd.to_timedelta(df["duration"])
        return cls(df)


#####
# Misc. module functions
#####


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


def reconcile_hypnograms(h1: pd.DataFrame, h2: pd.DataFrame) -> pd.DataFrame:
    """Combine two hypnograms such that any conflicts are resolved in favor of h1."""
    return utils.reconcile_labeled_intervals(
        h1, h2, "start_time", "end_time", "duration"
    )


def _check_datetime(dt):
    """Check that something is a valid datetime."""
    if is_datetime64_any_dtype(dt) or isinstance(dt, pd.Timestamp):
        return dt

    # Because pd.to_datetime will accept strings that don't contain a date, we have to check ourselves.
    if isinstance(dt, str):
        try:
            pd.core.tools.times.to_time(dt)
            warnings.warn(
                f"{dt} doesn't appear to include a date. Maybe you wanted `keep_between_time`?"
            )
        except ValueError:
            pass
        return pd.to_datetime(dt)

    raise ValueError("Unexpected datetime type.")


def _check_time(t):
    """Check that something is a valid time of day (e.g. 10:00:00, without a date)."""
    if isinstance(t, datetime.time):
        return t

    if isinstance(t, str):
        try:
            return pd.core.tools.times.to_time(t)
        except ValueError:
            raise ValueError(
                f"{t} could not be converted to a dateless time of day. Maybe you wanted `keep_between_datetime`?"
            )

    raise ValueError("Unexpected time of day type.")


def remove_subsumed(df: pd.DataFrame) -> pd.DataFrame:
    """Remove bouts that are wholly subsumed by other bouts."""
    if not {"state", "start_time", "end_time", "duration"}.issubset(df):
        raise AttributeError(
            "Required columns `state`, `start_time`, `end_time`, and `duration` are not present."
        )
    if not all(df["start_time"] <= df["end_time"]):
        raise ValueError("Not all start times precede end times.")

    keep = list()
    for i in range(len(df)):
        contains = (df["start_time"] <= df.iloc[i]["start_time"]) & (
            df["end_time"] >= df.iloc[i]["end_time"]
        )
        if not contains.sum() > 1:
            keep.append(i)

    return df.iloc[keep].reset_index(drop=True)


def condense(df: pd.DataFrame, tolerance) -> pd.DataFrame:
    """Condense a Hypnogram so that consecutive bouts with matching states,
    separated by a small gap, are combined into a single hypnogram entry.

    Parameters:
    -----------
    tolerance:
        Gaps between consecutive bouts with matching states, less than or equal to this value, are incorporated as that state.

    Note that this will also condense if consecutive matching bouts overlap!
    But, of course, it will not touch overlaps between non-matching states! This is desired behavior!
    """
    if not {"state", "start_time", "end_time", "duration"}.issubset(df):
        raise AttributeError(
            "Required columns `state`, `start_time`, `end_time`, and `duration` are not present."
        )
    if not all(df["start_time"] <= df["end_time"]):
        raise ValueError("Not all start times precede end times.")
    if not df["start_time"].is_monotonic_increasing:
        raise ValueError("Hypnogram start times are not monotonically increasing.")
    if not df["end_time"].is_monotonic_increasing:
        raise ValueError("Hypnogram end times are not monotonically increasing.")

    result = list()
    i = 0
    while i < len(df):
        curr = df.iloc[i].to_dict()
        j = i + 1
        while (
            (j < len(df))
            and (df.iloc[j]["state"] == curr["state"])
            and (df.iloc[j]["start_time"] - curr["end_time"] <= tolerance)
        ):
            curr = {
                "state": curr["state"],
                "start_time": curr["start_time"],
                "end_time": df.iloc[j]["end_time"],
                "duration": df.iloc[j]["end_time"] - curr["start_time"],
            }
            j += 1
        result.append(curr)
        i = j

    return pd.DataFrame(result)


def _trim_overlap(df: pd.DataFrame) -> pd.DataFrame:
    """Helper function for `trim_overlap`. Does the actual reconciliation of bouts."""
    if len(df) <= 1:
        return df
    df = df.sort_values("duration", ascending=False)
    longestBout = df.iloc[0]
    rest = df.iloc[1:].copy()  # copy() just to avoid SettingithCopyWarning
    trimTail = (rest["start_time"] < longestBout["start_time"]) & (
        rest["end_time"] > longestBout["start_time"]
    )
    trimHead = (rest["end_time"] > longestBout["end_time"]) & (
        rest["start_time"] < longestBout["end_time"]
    )
    rest.loc[trimTail, "end_time"] = longestBout["start_time"]
    rest.loc[trimHead, "start_time"] = longestBout["end_time"]
    rest["duration"] = rest["end_time"] - rest["start_time"]

    return (
        pd.concat([longestBout.to_frame().T, _trim_overlap(rest)])
        .sort_values("start_time")
        .reset_index(drop=True)
    )


def trim_overlap(df: pd.DataFrame) -> pd.DataFrame:
    """Trim bouts so that they do not overlap, resolving conflicts in favor of longer bouts.

    Works by finding sets of bouts such that each bout in a set overlaps with at least one other bout in that set.
    For example, in {(0, 2), (1, 4}, (3, 5)}, (0, 2) and (3, 5) do not directly overlap, but both share overlap with (1, 4).
    Once a set is found, take the longest bout within that set, trim others to fit it, repeat with the next longest bout, and so on."""
    if not {"state", "start_time", "end_time", "duration"}.issubset(df):
        raise AttributeError(
            "Required columns `state`, `start_time`, `end_time`, and `duration` are not present."
        )
    if not all(df["start_time"] <= df["end_time"]):
        raise ValueError("Not all start times precede end times.")
    if not df["start_time"].is_monotonic_increasing:
        raise ValueError("Hypnogram start times are not monotonically increasing.")

    result = list()
    i = 0
    while i < len(df):
        end = df.iloc[i]["end_time"]
        j = i + 1
        while (j < len(df)) and (df.iloc[j]["start_time"] < end):
            end = df.iloc[j]["end_time"]
            j += 1
        result.append(_trim_overlap(df.iloc[i:j]))
        i = j

    return pd.concat(result, ignore_index=True)


def get_gaps(df: pd.DataFrame, longerThan) -> list[dict]:
    """Get all unscored gaps in the hypnogram.

    Parameters:
    -----------
    longerThan:
        Only get gaps greater than a given duration.

    Returns:
    --------
    gaps: list of dict
        Each gap detected, with start_time, end_time, and duration.
    """
    if not {"state", "start_time", "end_time", "duration"}.issubset(df):
        raise AttributeError(
            "Required columns `state`, `start_time`, `end_time`, and `duration` are not present."
        )
    if not all(df["start_time"] <= df["end_time"]):
        raise ValueError("Not all start times precede end times.")
    if not df["start_time"].is_monotonic_increasing:
        raise ValueError("Hypnogram start times are not monotonically increasing.")
    if not df["end_time"].is_monotonic_increasing:
        raise ValueError("Hypnogram end times are not monotonically increasing.")

    gaps = list()
    for i in range(len(df) - 1):
        current_bout_end = df.iloc[i].end_time
        next_bout_start = df.iloc[i + 1].start_time
        gap = next_bout_start - current_bout_end
        if gap > longerThan:
            gaps.append(
                dict(
                    start_time=current_bout_end,
                    end_time=next_bout_start,
                    duration=gap,
                )
            )

    return gaps


def fill_gaps(df: pd.DataFrame, longerThan, **kwargs) -> pd.DataFrame:
    """Fill all unscored gaps in the hypnogram.

    Parameters:
    -----------
    longerThan:
        Only fill gaps greater than a given duration.
    kwargs:
        See pd.DataFrame.fillna()

    Examples:
    ---------
    To mark all gaps >1s as "Unscored", then fill all smaller gaps in with the preceeding state:
    df = fill_gaps(df, longerThan=1, value="Unscored")
    df = fill_gaps(df, longerthan=0, method="ffill)
    """
    if not {"state", "start_time", "end_time", "duration"}.issubset(df):
        raise AttributeError(
            "Required columns `state`, `start_time`, `end_time`, and `duration` are not present."
        )
    if not all(df["start_time"] <= df["end_time"]):
        raise ValueError("Not all start times precede end times.")
    if not df["start_time"].is_monotonic_increasing:
        raise ValueError("Hypnogram start times are not monotonically increasing.")
    if not df["end_time"].is_monotonic_increasing:
        raise ValueError("Hypnogram end times are not monotonically increasing.")

    gaps = get_gaps(df, longerThan)
    for gap in gaps:
        gap.update({"state": np.nan})
    return (
        pd.concat([df, pd.DataFrame.from_records(gaps)])
        .sort_values("start_time", ignore_index=True)
        .fillna(**kwargs)
    )


def trim_hypnogram(df: pd.DataFrame, start, end) -> pd.DataFrame:
    """Trim a hypnogram to start and end within a specified time range.
    Actually will truncate bouts if they extend beyond the range."""
    if not {"state", "start_time", "end_time", "duration"}.issubset(df):
        raise AttributeError(
            "Required columns `state`, `start_time`, `end_time`, and `duration` are not present."
        )
    if not all(df["start_time"] <= df["end_time"]):
        raise ValueError("Not all start times precede end times.")

    df = df.copy()
    starts_before = df["start_time"] < start
    df.loc[starts_before, "start_time"] = start
    ends_after = df["end_time"] > end
    df.loc[ends_after, "end_time"] = end
    ends_before = df["end_time"] <= start
    df = df[~ends_before]
    starts_after = df["start_time"] >= end
    df = df[~starts_after]
    df["duration"] = df["end_time"] - df["start_time"]
    return df.reset_index(drop=True)


def clean(df: pd.DataFrame, condenseTol, missingDataTol, zero) -> pd.DataFrame:
    df = remove_subsumed(df)
    df = condense(df, condenseTol)
    df = trim_overlap(df)
    df = fill_gaps(df, missingDataTol, value="NoData")
    df = fill_gaps(df, longerThan=zero, method="ffill")
    return condense(df, tolerance=condenseTol)
