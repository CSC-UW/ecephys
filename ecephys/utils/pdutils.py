import numpy as np
import pandas as pd
import xarray as xr
from pandas.api.types import is_datetime64_ns_dtype
from pathlib import Path
from typing import Union, Optional
from . import xrutils

Pathlike = Union[Path, str]
ArrayLike = Union[np.array, list]


def write_htsv(df: pd.DataFrame, file: Pathlike):
    assert Path(file).suffix == ".htsv", "File must use extension .htsv"
    Path(file).parent.mkdir(
        parents=True, exist_ok=True
    )  # Make parent directories if they do not exist
    df.to_csv(file, sep="\t", header=True, index=(df.index.name is not None))


def read_htsv(file: Pathlike) -> pd.DataFrame:
    assert Path(file).suffix == ".htsv", "File must use extension .htsv"
    return pd.read_csv(file, sep="\t", header=0, float_precision='round_trip')


def store_pandas_netcdf(pd_obj: Union[pd.DataFrame, pd.Series], path: Pathlike):
    """Save a pandas object, including attrs, to a NetCDF file."""
    xr_obj = pd_obj.to_xarray()
    xr_obj.attrs = pd_obj.attrs
    xrutils.save_xarray_to_netcdf(xr_obj, path)


def read_pandas_netcdf(path: Pathlike):
    """Load a NetCDF file whose contents can be interpreted as a pandas object."""
    xr_obj = xr.load_dataset(path)
    pd_obj = xr_obj.to_pandas()
    pd_obj.attrs = xr_obj.attrs
    return pd_obj


def unnest_df(df: pd.DataFrame, col: str, reset_index: bool = False) -> pd.DataFrame:
    """
    References
    ----------
    [1] # https://stackoverflow.com/questions/21160134
    """
    import pandas as pd

    col_flat = pd.DataFrame(
        [[i, x] for i, y in df[col].apply(list).iteritems() for x in y],
        columns=["I", col],
    )
    col_flat = col_flat.set_index("I")
    df = df.drop(col, 1)
    df = df.merge(col_flat, left_index=True, right_index=True)
    if reset_index:
        df = df.reset_index(drop=True)
    return df


def store_df_h5(path: Pathlike, df: pd.DataFrame, **kwargs):
    """Store a DataFrame and a dictionary of metadata as HDF5.
    For conveience, `load_df_h5` can be used to read the data back.
    Probably works just fine to save non-DataFrames, but h5py may be safer.

    Parameters
    ----------
    filename: str
        The file create or write to.
    df: pandas.DataFrame
        The data to save
    **kwargs:
        Any metadata you wish to save with the primary data.

    Examples
    --------
    filename = '/path/to/myfile.h5'
    metadata = dict(foo='bar')
    store_df_h5(filename, df, **metadata)
    with pd.HDFStore(filename) as store:
        df, metadata = load_df_h5(store)

    References
    ----------
    [1] # https://stackoverflow.com/questions/29129095
    """
    # Create parent directories if they do not already exist.
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    store = pd.HDFStore(path)
    store.put("mydata", df)
    store.get_storer("mydata").attrs.metadata = kwargs
    store.close()


def load_df_h5(path: Pathlike) -> pd.DataFrame:
    """Read DataFrame and a dictionary of metadata as HDF5.
    Assumes data were saved using `store_df_h5`.

    Parameters
    ----------
    path: str or pathlib.Path
        The file create or write to.

    Returns
    -------
    df: pandas.DataFrame
        Metadata are saved in `df.attrs`
    """
    with pd.HDFStore(path) as store:
        df = store["mydata"]
        metadata = store.get_storer("mydata").attrs.metadata

    return add_attrs(df, **metadata)


def add_attrs(obj, **kwargs):
    for key in kwargs:
        obj.attrs[key] = kwargs[key]

    return obj


def dataframe_abs(df: pd.DataFrame) -> pd.DataFrame:
    "Take the absolute value of all numeric columns in a dataframe."
    _df = df.copy()
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            _df[col] = df[col].abs()


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


def dt_series_to_seconds(
    dt_series: pd.Series, t0: Optional[np.datetime64] = None
) -> np.ndarray:
    assert is_datetime64_ns_dtype(dt_series), "dt_series must be datetime64[ns] series"
    if t0 is None:
        t0 = dt_series.min()
    assert isinstance(t0, pd.Timestamp), "t0 must be datetime64[ns]"

    return (dt_series - t0).dt.total_seconds().values


def get_gaps(df, t1_colname: str = "start_time", t2_colname: str = "end_time", min_gap_duration_sec: float = 0):
    gaps = pd.DataFrame({
            t1_colname: df.iloc[:-1][t2_colname].values,
            t2_colname: df.iloc[1:][t1_colname].values,
    })
    gaps["duration"] = gaps[t2_colname] - gaps[t1_colname]
    return gaps[gaps["duration"] > min_gap_duration_sec]


def get_edges_start_end_samples_df(state_vector: ArrayLike):
    """Return df with `state`, `start_frame`, `end_frame` cols from array of states."""
    edges = np.where(state_vector[1:] != state_vector[0:-1])
    left_edges = np.insert(edges, 0, -1) + 1
    right_edges = np.append(edges, len(state_vector) - 1) + 1
    df = pd.DataFrame(
        {
            f"state": np.array(state_vector)[left_edges.astype(int)],
            f"start_frame": left_edges,
            f"end_frame": right_edges,
        }
    )
    return df


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

    if not (delta in df1.columns):
        df1[delta] = df1[hi] - df1[lo]
    if not (delta in df2.columns):
        df2[delta] = df2[hi] - df2[lo]

    for index, row in df1.iterrows():
        # If df2 contains any interval exactly equivalent to this one, drop it.
        identical_intervals = (df2[lo] == row[lo]) & (df2[hi] == row[hi])
        if any(identical_intervals):
            assert (
                sum(identical_intervals) == 1
            ), "More than one interval in df2 is identical to an interval found in df1. Is df2 well formed?"
            df2 = df2[~identical_intervals]

        # If df2 contains any intervals wholly contained by this one, drop them.
        sub_intervals = (df2[lo] >= row[lo]) & (df2[hi] <= row[hi])
        if any(sub_intervals):
            df2 = df2[~sub_intervals]

        # If df2 contains any interval that whole contains this one, split it into preceeding (left) and succeeding (right) intervals.
        super_intervals = (df2[lo] <= row[lo]) & (df2[hi] >= row[hi])
        if any(super_intervals):
            assert (
                sum(super_intervals) == 1
            ), "More than one interval in df2 wholly contains an interval found in df1. Is df2 well formed?"
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
            assert (
                sum(left_intervals) == 1
            ), "More than one interval in h2 overlaps the start of an interval found in h1. Is h2 well formed?"
            left_interval = df2[left_intervals].copy()
            left_interval[hi] = row[lo]
            left_interval[delta] = left_interval[hi] - left_interval[lo]
            df2[left_intervals] = left_interval

        # If df2 contains any interval that overlaps the endof this interval, adjust its start time.
        right_intervals = (
            (df2[lo] > row[lo]) & (df2[lo] < row[hi]) & (df2[hi] > row[hi])
        )
        if any(right_intervals):
            assert (
                sum(right_intervals) == 1
            ), "More than one interval in df2 overlaps the end of an interval found in df1. Is df2 well formed?"
            right_interval = df2[right_intervals].copy()
            right_interval[lo] = row[hi]
            right_interval[delta] = right_interval[hi] - right_interval[lo]
            df2[right_intervals] = right_interval

    result = pd.concat([df2, df1]).sort_values(lo).reset_index(drop=True)
    return result.loc[~(result[delta] == 0)]
