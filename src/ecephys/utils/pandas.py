import warnings
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats._result_classes import PearsonRResult

Pathlike = Union[Path, str]


def write_htsv(df: pd.DataFrame, file: Pathlike):
    assert Path(file).suffix == ".htsv", "File must use extension .htsv"
    Path(file).parent.mkdir(
        parents=True, exist_ok=True
    )  # Make parent directories if they do not exist
    df.to_csv(file, sep="\t", header=True, index=(df.index.name is not None))


def read_htsv(file: Pathlike) -> pd.DataFrame:
    assert Path(file).suffix == ".htsv", "File must use extension .htsv"
    return pd.read_csv(file, sep="\t", header=0, float_precision="round_trip")


def get_grouped_ecdf(df: pd.DataFrame, col: str, group_var: str) -> pd.DataFrame:
    "Get ECDFs in arbitary groups for plotting using sns.lineplot."
    ecdfs = list()
    for group_name, dat in df.groupby(group_var):
        dat_sorted = np.sort(dat[col])
        ecdf = 1.0 * np.arange(len(dat[col])) / (len(dat[col]) - 1)
        ecdfs.append(
            pd.DataFrame({col: dat_sorted, "ecdf": ecdf, group_var: group_name})
        )

    return pd.concat(ecdfs)


def reconcile_labeled_intervals(
    df1: pd.DataFrame, df2: pd.DataFrame, lo: str, hi: str, delta: str = "delta"
) -> pd.DataFrame:
    """Combine two dataframes of labeled intervals, such that any conflicts (i.e.
    overlapping intervals with different labels in df1 and df2) are resolved in favor
    of df1. It is expected that there is some other column, say `label`, that contains
    the label. But we do not need to know that column's name here. Keep in mind that
    interval endpoints are all considered open, so intervals (a, b) and (b, c) do NOT
    overlap.

    This function is useful for reconciling hypnograms, anatomical structure tables,
    data selection tables, etc.
    """
    df1 = df1.copy().sort_values(lo)
    df2 = df2.copy().sort_values(lo)

    if delta not in df1.columns:
        df1[delta] = df1[hi] - df1[lo]
    if delta not in df2.columns:
        df2[delta] = df2[hi] - df2[lo]

    for index, row in df1.iterrows():
        # If df2 contains any interval exactly equivalent to this one, drop it.
        identical_intervals = (df2[lo] == row[lo]) & (df2[hi] == row[hi])
        if any(identical_intervals):
            assert sum(identical_intervals) == 1, (
                "More than one interval in df2 is identical to an interval found in df1. Is df2 well formed?"
            )
            df2 = df2[~identical_intervals]

        # If df2 contains any intervals wholly contained by this one, drop them.
        sub_intervals = (df2[lo] >= row[lo]) & (df2[hi] <= row[hi])
        if any(sub_intervals):
            df2 = df2[~sub_intervals]

        # If df2 contains any interval that whole contains this one, split it into preceeding (left) and succeeding (right) intervals.
        super_intervals = (df2[lo] <= row[lo]) & (df2[hi] >= row[hi])
        if any(super_intervals):
            assert sum(super_intervals) == 1, (
                "More than one interval in df2 wholly contains an interval found in df1. Is df2 well formed?"
            )
            super_interval = df2[super_intervals]
            left_interval = super_interval.copy()
            left_interval[hi] = row[lo]
            left_interval[delta] = left_interval[hi] - left_interval[lo]
            right_interval = super_interval.copy()
            right_interval[lo] = row[hi]
            right_interval[delta] = right_interval[hi] - right_interval[lo]
            df2 = df2[~super_intervals]
            df2 = (
                pd.concat([df2, left_interval, right_interval])
                .sort_values(lo)
                .reset_index(drop=True)
            )

        # If df2 contains any interval that overlaps the start of this interval, truncate it.
        left_intervals = (df2[lo] < row[lo]) & (df2[hi] > row[lo]) & (df2[hi] < row[hi])
        if any(left_intervals):
            assert sum(left_intervals) == 1, (
                "More than one interval in h2 overlaps the start of an interval found in h1. Is h2 well formed?"
            )
            left_interval = df2[left_intervals].copy()
            left_interval[hi] = row[lo]
            left_interval[delta] = left_interval[hi] - left_interval[lo]
            df2[left_intervals] = left_interval

        # If df2 contains any interval that overlaps the end of this interval, adjust its start time.
        right_intervals = (
            (df2[lo] > row[lo]) & (df2[lo] < row[hi]) & (df2[hi] > row[hi])
        )
        if any(right_intervals):
            assert sum(right_intervals) == 1, (
                "More than one interval in df2 overlaps the end of an interval found in df1. Is df2 well formed?"
            )
            right_interval = df2[right_intervals].copy()
            right_interval[lo] = row[hi]
            right_interval[delta] = right_interval[hi] - right_interval[lo]
            df2[right_intervals] = right_interval

    result = (
        df1.copy() if df2.empty else df2.copy() if df1.empty else pd.concat([df2, df1])
    )  # Concatenate, handling possibly empty dataframes
    # Sort by (lo, hi), not lo alone: when two intervals share a start time (e.g.
    # sub-millisecond NoData/Artifact slivers produced by boundary reconciliation),
    # sorting by lo only leaves their relative order to a non-stable sort, which can
    # place a longer interval before a shorter one and yield non-monotonic end times.
    # FloatHypnogram validation requires monotonic end_time, so make the order total.
    result = result.sort_values([lo, hi]).reset_index(drop=True)
    return result.loc[~(result[delta] == 0)]


def pearsonr(sample1: pd.Series, sample2: pd.Series) -> PearsonRResult:
    """Just a thin wrapper around pearsonr that can handle nans.
    Just like the nan_policy='omit' option in scipy.stats.spearmanr
    """
    warnings.warn(
        "This function has been marked for deprecation. If you want it kept, remove this warning.",
        DeprecationWarning,
    )
    if sample1.isna().any() or sample2.isna().any():
        print(
            "Pearson's r is not defined for unequal sample sizes. Dropping observations with missing samples."
        )
        df = pd.concat([sample1, sample2], axis=1).dropna()
        sample1 = df.iloc[:, 0]
        sample2 = df.iloc[:, 1]
    return stats.pearsonr(sample1, sample2)


def cohens_d(sample1: pd.Series, sample2: pd.Series) -> float:
    warnings.warn(
        "This function has been marked for deprecation. If you want it kept, remove this warning.",
        DeprecationWarning,
    )
    if sample1.isna().any() or sample2.isna().any():
        print(
            "Cohen's D is not defined for unequal sample sizes. Dropping observations with missing samples."
        )
        df = pd.concat([sample1, sample2], axis=1).dropna()
        sample1 = df.iloc[:, 0]
        sample2 = df.iloc[:, 1]
    return abs(sample1.mean() - sample2.mean()) / (sample1 - sample2).std()


def mutual_labeled_intervals(
    *dfs: pd.DataFrame,
    lo: str,
    hi: str,
    delta: str,
    label: str,
    validate_input: bool = True,
) -> pd.DataFrame:
    """
    Combine any number of dataframes of labeled intervals, such that only intervals of agreement
    (intervals with the same labels in all dataframes) are retained.

    Any interval of disagreement (intervals with different labels in any dataframe, or
    intervals for which only some dataframes contain a label) are dropped.

    Keep in mind that interval endpoints are all considered open,
    so intervals (a, b) and (b, c) do NOT overlap.

    This function assumes that:
    - Intervals are sorted by start time (lo) within each label group.
    - Intervals do not overlap within a dataframe for a given label (i.e., intervals
      for the same label are disjoint and sorted).
    These assumptions allow us to use a merge-like sweep (O(n1 + n2 + ... + nk)) to find the
    mutual intervals, rather than the naive nested loop approach (O(n1 * n2 * ... * nk)).

    This function is useful for finding agreement between hypnograms,
    anatomical structure tables, data selection tables, etc.

    Parameters:
    -----------
    *dfs: pd.DataFrame
        Any number of dataframes of labeled intervals.
    lo: str
        The column name of the interval start (e.g. "start_time").
    hi: str
        The column name of the interval end (e.g. "end_time").
    delta: str
        The column name of the interval delta (e.g. "duration").
    label: str
        The column name of the interval label (e.g. "state").
    validate_input: bool
        If True, check that the input dataframes are valid.

    Returns:
    --------
    pd.DataFrame
        DataFrame of intervals where all input dataframes agree on the label.
    """
    if len(dfs) < 2:
        raise ValueError("At least two dataframes are required.")
    for idx, df in enumerate(dfs):
        if validate_input and not check_disjoint_sorted(df, lo, hi, label):
            raise ValueError(
                f"df{idx + 1} intervals are not sorted and disjoint within each label group."
            )

    # Only process labels present in all dataframes
    common_labels = set(dfs[0][label])
    for df in dfs[1:]:
        common_labels &= set(df[label])

    results = []
    for lbl in common_labels:
        intervals = [
            df[df[label] == lbl].sort_values(lo).reset_index(drop=True) for df in dfs
        ]
        # Initialize pointers for each dataframe
        idxs = [0] * len(intervals)
        while all(idx < len(intervals[i]) for i, idx in enumerate(idxs)):
            starts = [intervals[i].loc[idxs[i], lo] for i in range(len(intervals))]
            ends = [intervals[i].loc[idxs[i], hi] for i in range(len(intervals))]
            start = max(starts)
            end = min(ends)
            if start < end:
                results.append({label: lbl, lo: start, hi: end, delta: end - start})
            # Advance the pointer(s) with the earliest end
            min_end = min(ends)
            for i in range(len(idxs)):
                if ends[i] == min_end:
                    idxs[i] += 1
    if results:
        return pd.DataFrame(results).sort_values(lo).reset_index(drop=True)
    else:
        return pd.DataFrame(columns=[label, lo, hi, delta])


def check_disjoint_sorted(df: pd.DataFrame, lo: str, hi: str, label: str) -> bool:
    """
    Check that, for each label, intervals are sorted by `lo` and do not overlap.
    Returns True if all label groups are sorted and disjoint, False otherwise.
    """
    for lbl, group in df.groupby(label):
        starts = group[lo].values
        ends = group[hi].values
        # Check sorted
        if not np.all(starts[:-1] <= starts[1:]):
            return False
        # Check disjoint: end of previous <= start of next (open intervals)
        if not np.all(ends[:-1] <= starts[1:]):
            return False
    return True
