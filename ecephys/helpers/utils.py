import numpy as np


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


# https://stackoverflow.com/questions/21160134/flatten-a-column-with-value-of-type-list-while-duplicating-the-other-columns-va
def unnest_df(df, col, reset_index=False):
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
