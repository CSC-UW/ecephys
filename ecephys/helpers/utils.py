import numpy as np

def next_power_of_2(x):
    """Return the smallest power of 2 greater than or equal to x."""
    return 1<<(np.int(x)-1).bit_length()

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
