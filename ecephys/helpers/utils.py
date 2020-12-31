import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation


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
    store = pd.HDFStore(filename)
    store.put("mydata", df)
    store.get_storer("mydata").attrs.metadata = kwargs
    store.close()


def load_df_h5(store):
    """Read DataFrame and a dictionary of metadata as HDF5.
    Assumes data were saved using `store_df_h5`.

    Parameters
    ----------
    store: pandas.HDFStore
        The file create or write to.

    Returns
    -------
    df: pandas.DataFrame
    metadata: dict

    Examples
    --------
    with pd.HDFStore(filename) as store:
        df, metadata = load_df_h5(store)
    """
    df = store["mydata"]
    metadata = store.get_storer("mydata").attrs.metadata
    return df, metadata


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