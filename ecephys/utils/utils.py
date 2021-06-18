import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation
from pathlib import Path
from collections.abc import Iterable

# https://stackoverflow.com/questions/2158395/flatten-an-irregular-list-of-lists
def flatten(l):
    for el in l:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el


def if_none(x, default):
    if x is None:
        return x
    else:
        return default


def next_power_of_2(x):
    """Return the smallest power of 2 greater than or equal to x."""
    return 1 << (np.int(x) - 1).bit_length()


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


def nrows(x):
    return x.shape[0]


def ncols(x):
    return x.shape[1]


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


def find_nearest(array, value):
    """ https://stackoverflow.com/questions/2566412 """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def add_attrs(obj, **kwargs):
    for key in kwargs:
        obj.attrs[key] = kwargs[key]

    return obj


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