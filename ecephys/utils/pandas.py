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
    """Combine two dataframes of labeled intervals, such that any conflicts (i.e. overlapping intervals with different labels in df1 and df2) are resolved in favor of df1.
    It is expected that there is some other column, say `label`, that contains the label. But we do not need to know that columns name here.
    Keep in mind that interval endpoints are all considered open, so intervals (a, b) and (b, c) do NOT overlap.

    This function is useful for reconciling hypnograms, anatomical structure tables, data selection tables, etc.
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

        # If df2 contains any interval that overlaps the endof this interval, adjust its start time.
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
    result = result.sort_values(lo).reset_index(drop=True)  # Sort and reset index
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
