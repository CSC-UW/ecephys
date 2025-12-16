import dask.array as da
import numpy as np
import itertools as it


def get_dask_chunk_bounds(dsk: da.Array, axis=0, pairwise=False):
    """Includes (0, end)"""
    chunks = dsk.chunks[axis]
    bounds = np.cumsum((0,) + chunks)
    if pairwise:
        bounds = it.pairwise(bounds)
    return bounds
