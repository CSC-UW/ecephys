from collections.abc import Iterable
from functools import reduce
import itertools as it
import json
import logging
import pathlib

import heapq
import numpy as np
from rich.console import Console
from scipy.stats import median_abs_deviation
import sortednp

logger = logging.getLogger(__name__)
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


def clip_outliers(x, method="mad", n_devs=6):
    if method == "mad":
        center = np.median(x)
        dev = median_abs_deviation(x)
    elif method == "std":
        center = np.mean(x)
        dev = np.std(x)
    else:
        raise ValueError(f"Unrecognized method: {dev}")

    hi = center + n_devs * dev
    lo = center - n_devs * dev

    x[x > hi] = hi
    x[x < lo] = lo
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


def pairwise(iterable):
    "s -> (s0, s1), (s1, s2), (s2, s3), ..."
    a, b = it.tee(iterable)
    next(b, None)
    return zip(a, b)


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


# https://medium.com/@amitrajit_bose/merge-k-sorted-arrays-6f9427661e67
def kway_sortednp_merge(arrays):
    if len(arrays) == 0:
        return np.array([])
    elif len(arrays) == 1:
        return arrays[0]
    elif len(arrays) == 2:
        return sortednp.merge(arrays[0], arrays[1])
    # return sortednp.kway_merge(arrays)

    lists = [a.tolist() for a in arrays]
    final_list = []
    heap = [(mylst[0], i, 0) for i, mylst in enumerate(lists) if mylst]
    heapq.heapify(heap)

    while heap:
        val, list_ind, element_ind = heapq.heappop(heap)
        final_list.append(val)
        if element_ind + 1 < len(lists[list_ind]):
            next_tuple = (lists[list_ind][element_ind + 1], list_ind, element_ind + 1)
            heapq.heappush(heap, next_tuple)
    return np.array(final_list)


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


# -------------------- Miscellaneous --------------------


def hotfix_times(times: np.ndarray) -> bool:
    """Given an array of times that should be increasingly monotonically, find decreases and force the offending timestamps to no-change.
    This is horribly inefficient. If you actually expect lots of hotfixes, do better."""
    cum_n_lzd = 0
    n_lzd = -1
    n_iters = 0
    while n_lzd != 0:
        diffs = np.diff(times)
        # Find zero diffs, generally NOT an issue
        zd = diffs == 0
        n_zd = zd.sum()
        if n_zd > 0:
            logger.debug(f"Found {n_zd} zero diffs in timestamps on pass {n_iters}.")
        # Find less-than-zero diffs, generally YES an issue
        lzd = diffs < 0
        n_lzd = lzd.sum()
        cum_n_lzd += n_lzd
        if n_lzd > 0:
            logger.debug(
                f"Found {n_lzd} less-than-zero diffs in timestamps on pass {n_iters}. Hotfixing..."
            )
            lzd_ixs = np.where(lzd)[0]
            # Problematic were between times[lzd_ixs] and times[lzd_ixs + 1]
            times[lzd_ixs] = times[lzd_ixs + 1]  # Turn these into zero diffs
        n_iters += 1

    if cum_n_lzd > 0:
        logger.warning(
            f"Hotfixed {cum_n_lzd} less-than-zero diffs in timestamps over {n_iters + 1} passes."
        )
    return cum_n_lzd


def write_json(obj: dict, path: pathlib.Path, indent: int = 4):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=indent, default=str)
