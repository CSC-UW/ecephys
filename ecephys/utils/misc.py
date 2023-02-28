import numpy as np
import itertools as it
from scipy.stats import median_abs_deviation
from collections.abc import Iterable
from functools import reduce
from rich.console import Console

_console = Console()

def warn(msg):
    _console.print(f"Warning: {msg}", style="bright_yellow")


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


def discard_outliers(x, n_mad=6):
    mad = median_abs_deviation(x)
    threshold = np.median(x) + n_mad * mad
    return x[x <= threshold]


def replace_outliers(x, n_mad=6, fill_value=np.nan):
    mad = median_abs_deviation(x)
    threshold = np.median(x) + n_mad * mad
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


# -------------------- Iterator utilities --------------------


def roundrobin(*iterables):
    "roundrobin('ABC', 'D', 'EF') --> A D E B F C"
    # Recipe credited to George Sakkis
    num_active = len(iterables)
    nexts = it.cycle(iter(it).__next__ for it in iterables)
    while num_active:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            # Remove the iterator we just exhausted from the cycle.
            num_active -= 1
            nexts = it.cycle(it.islice(nexts, num_active))


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