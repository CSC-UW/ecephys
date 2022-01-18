import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_ns_dtype
from scipy.stats import median_abs_deviation
from pathlib import Path
from collections.abc import Iterable
from functools import reduce


# -------------------- Filesystem utilities --------------------

# Avoid PermissionError with shutil.copytree on NAS smb share
# TODO: Move to wisc-specific
def system_copy(src, dst):
    """Copy using `cp -r src dst` system call."""
    import subprocess

    subprocess.call(["cp", "-r", str(src), str(dst)])


# -------------------- Pattern utilities --------------------


def if_none(x, default):
    if x is None:
        return x
    else:
        return default


# -------------------- Stats & Math utilities --------------------


def next_power_of_2(x):
    """Return the smallest power of 2 greater than or equal to x."""
    return 1 << (np.int(x) - 1).bit_length()


def discard_outliers(x):
    mad = median_abs_deviation(x)
    threshold = np.median(x) + 6 * mad
    return x[x <= threshold]


def replace_outliers(x, fill_value=np.nan):
    mad = median_abs_deviation(x)
    threshold = np.median(x) + 6 * mad
    x[x > threshold] = fill_value
    return x


def zscore_to_value(data, z):
    return z * data.std() + data.mean()


# -------------------- List utilities --------------------


def remove_duplicates(l):
    """Given a list l, remove duplicate items while preserving order."""
    return list(dict.fromkeys(l))


# https://stackoverflow.com/questions/2158395/flatten-an-irregular-list-of-lists
def flatten(l):
    for el in l:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el


# -------------------- Dict utilities --------------------


def item_intersection(l):
    """Give a list of dictionaries l, keep only the intersection of the key-value pairs.

    Examples
    ========
    foo = dict(a=1, b=2, c=3)
    bar = dict(a=10, b=2, c=8)
    baz = dict(a=10, b=2, c=3)
    item_intersection([foo, bar, baz])
    >> {'b': 2}
    """
    return reduce(lambda x, y: dict(x.items() & y.items()), l)


# -------------------- DataFrame utilities --------------------


def unnest_df(df, col, reset_index=False):
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


def store_df_h5(filename, df, **kwargs):
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
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    store = pd.HDFStore(filename)
    store.put("mydata", df)
    store.get_storer("mydata").attrs.metadata = kwargs
    store.close()


def load_df_h5(path):
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


def dataframe_abs(df):
    "Take the absolute value of all numeric columns in a dataframe."
    _df = df.copy()
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            _df[col] = df[col].abs()


def get_grouped_ecdf(df, col, group_var):
    "Get ECDFs in arbitary groups for plotting using sns.lineplot."
    ecdfs = list()
    for group_name, dat in df.groupby(group_var):
        dat_sorted = np.sort(dat[col])
        ecdf = 1.0 * np.arange(len(dat[col])) / (len(dat[col]) - 1)
        ecdfs.append(
            pd.DataFrame({col: dat_sorted, "ecdf": ecdf, group_var: group_name})
        )

    return pd.concat(ecdfs)


def dt_series_to_seconds(dt_series, t0=None):
    assert is_datetime64_ns_dtype(dt_series), "dt_series must be datetime64[ns] series"
    if t0 is None:
        t0 = dt_series.min()
    assert isinstance(t0, pd.Timestamp), "t0 must be datetime64[ns]"

    return (dt_series - t0).dt.total_seconds().values


def get_epocs(df, col, t):
    edges = np.where(np.diff(df[col]))
    left_edges = np.insert(edges, 0, -1) + 1
    right_edges = np.append(edges, len(df) - 1)
    epocs = pd.DataFrame(
        {
            f"start_{t}": df.iloc[left_edges][t].values,
            f"end_{t}": df.iloc[right_edges][t].values,
            col: df.iloc[left_edges][col].values,
        }
    ).set_index([f"start_{t}", f"end_{t}"])

    for left, right, val in zip(left_edges, right_edges, epocs[col]):
        epoc = df.iloc[left:right]
        assert all(epoc[col].values == val), f"{col} should not change during an epoch."

    return epocs


# -------------------- Array utilities --------------------

# Get rid of this
def nrows(x):
    return x.shape[0]


# Get rid of this
def ncols(x):
    return x.shape[1]


def find_nearest(array, value, tie_select="first"):
    """Index of element in array nearest to value.

    Return either first or last value if ties"""
    array = np.asarray(array)
    a = np.abs(array - value)
    if tie_select == "first":
        return a.argmin()
    elif tie_select == "last":
        # reverse array to find last occurence
        b = a[::-1]
        return len(b) - np.argmin(b) - 1
    else:
        raise ValueError()


def round_to_values(arr, values):
    "Round each value in arr to the nearest value in values."
    f = np.vectorize(lambda x: find_nearest(values, x))
    idx = np.apply_along_axis(f, 0, arr)
    return values[idx]


def array_where(a1, a2):
    """Get indices into a2 of each element in a1"""
    return np.apply_along_axis(np.vectorize(lambda x: np.where(a2 == x)[0]), 0, a1)


def all_arrays_equal(iterator):
    """Check if all arrays in the iterator are equal."""
    try:
        iterator = iter(iterator)
        first = next(iterator)
        return all(np.array_equal(first, rest) for rest in iterator)
    except StopIteration:
        return True


def all_equal(iterator):
    """Check if all items in an un-nested array are equal."""
    try:
        iterator = iter(iterator)
        first = next(iterator)
        return all(first == rest for rest in iterator)
    except StopIteration:
        return True


def shift_array(arr, num, fill_value=np.nan):
    """Shift arr by num places (can be postive, negative, or zero),
    filling places where values are shifted out with a set value. Fast!"""
    # Does not shift in place. Allocates new array for result.
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result


def get_values_around(arr, val, num):
    """Get num values from arr, centered on the unique occurence of val."""
    assert num % 2, "Must get an odd number of channels."
    result = np.argwhere(arr == val)
    assert result.size == 1, "arr should contain val exactly once."
    idx = result[0].squeeze()
    first = idx - num // 2
    last = idx + num // 2 + 1

    assert first >= 0, "Requested values outside the bounds of your array."
    assert last < len(arr), "Requested values outside the bounds of your array."

    return arr[first:last]


# -------------------- 2D array utils --------------------


def roll_cols(M, rolls):
    """Roll columns of a 2D matrix independently."""
    # Rolls, rather than shifts
    # Does not roll in place
    rows, cols = np.ogrid[: M.shape[0], : M.shape[1]]
    rolls[rolls < 0] += M.shape[0]
    rows = rows - rolls[np.newaxis, :]
    return M[rows, cols]


def roll_rows(M, rolls):
    """Roll rows of a 2D matrix independently."""
    # Rolls, rather than shifts
    # Does not roll in place
    rows, cols = np.ogrid[: M.shape[0], : M.shape[1]]
    rolls[rolls < 0] += M.shape[1]
    cols = cols - rolls[:, np.newaxis]
    return M[rows, cols]


# TODO remove this function?
def shift_matrix(M, shifts, axis):
    """Shift rows or columns of a 2D matrix independently."""
    # This takes at least 35 mins for a a 384ch 2hr lfp. Useless.
    # axis=0 shifts rows, positive shifts move rows right (towards higher indices)
    # axis=1 shifts cols, positive shifts move rows down (towards higher indices)
    # Does not shift in place
    f = lambda i: shift_array(M.take(i, axis=axis), shifts[i])
    idx = np.arange(M.shape[axis])
    return np.stack([f(i) for i in idx], axis=axis)


def shift_cols(M, shifts, fill_value=np.nan):
    """Shift columns of a 2D matrix independently."""
    rows, cols = np.ogrid[: M.shape[0], : M.shape[1]]
    pos = np.zeros_like(shifts)
    neg = np.zeros_like(shifts)

    pos[shifts > 0] = shifts[shifts > 0]
    neg[shifts < 0] = shifts[shifts < 0]

    shifts[shifts < 0] += M.shape[0]

    posM = rows + pos[np.newaxis, :] - M.shape[0]
    negM = rows + neg[np.newaxis, :]
    rows = rows - shifts[np.newaxis, :]

    M[(posM >= 0) | (negM < 0)] = fill_value
    return M[rows, cols]


def shift_rows(M, shifts, fill_value=np.nan):
    """Shift rows of a 2D matrix independently,
    replacing shifted-out values with NaN."""
    # Shifts is an np.int64 array, where shifts[i] is the number
    # of shifts to apply to row i. Negative numbers inddicate shifts
    # towards lower indices, positive numbers towards higher indices.
    #
    # This takes about 6.5 minutes for a 384ch 2hr lfp.
    # It is surprisingly not that memory or CPU intensive.
    rows, cols = np.ogrid[: M.shape[0], : M.shape[1]]
    pos = np.zeros_like(shifts)
    neg = np.zeros_like(shifts)

    pos[shifts > 0] = shifts[shifts > 0]
    neg[shifts < 0] = shifts[shifts < 0]

    shifts[shifts < 0] += M.shape[1]

    posM = cols + pos[:, np.newaxis] - M.shape[1]
    negM = cols + neg[:, np.newaxis]
    cols = cols - shifts[:, np.newaxis]

    M[(posM >= 0) | (negM < 0)] = fill_value
    return M[rows, cols]


# -------------------- Generic algorithms --------------------


def get_disjoint_interval_intersections(arr1, arr2):
    """Given two 2-D arrays which represent intervals. Each 2-D array represents a list
    of intervals. Each list of intervals is disjoint and sorted in increasing order.
    Find the intersection or set of ranges that are common to both the lists.

    Examples
    ---------
    Input:
        arr1 = [(0, 4), (5, 10), (13, 20), (24, 25)]
        arr2 = [(1, 5), (8, 12), (15, 24), (25, 26)]
    Output:
        [(1, 4), (5, 5), (8, 10), (15, 20), (24, 24), (25, 25)]

    References
    ----------
    [1] https://www.geeksforgeeks.org/find-intersection-of-intervals-given-by-two-lists/
    """

    intersection = list()

    # i and j pointers for arr1
    # and arr2 respectively
    i = j = 0

    n = len(arr1)
    m = len(arr2)

    # Loop through all intervals unless one
    # of the interval gets exhausted
    while i < n and j < m:

        # Left bound for intersecting segment
        l = max(arr1[i][0], arr2[j][0])

        # Right bound for intersecting segment
        r = min(arr1[i][1], arr2[j][1])

        # If segment is valid print it
        if l <= r:
            intersection.append((l, r))

        # If i-th interval's right bound is
        # smaller increment i else increment j
        if arr1[i][1] < arr2[j][1]:
            i += 1
        else:
            j += 1

    return intersection


def get_interval_complements(intervals, start_time, end_time):
    """Get intervals complementary to those provided over a specified time range.

    Examples
    --------
    Input:
        intervals = [(0, 4), (5, 10), (13, 20), (24, 25)]
        start_time, end_time = (0, 30)
    Output:
        [(4, 5), (10, 13), (20, 24), (25, 30)]
    """

    intervals = np.asarray(intervals)
    complement = list()

    assert start_time < intervals.flatten()[0]
    assert end_time > intervals.flatten()[-1]

    edges = np.concatenate([[start_time], intervals.flatten(), [end_time]])
    it = iter(edges)
    for l in it:
        r = next(it)
        complement.append((l, r))

    return complement
