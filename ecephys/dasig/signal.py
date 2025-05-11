import dask.array as da

from ecephys import npsig


def hilbert(data: da.Array) -> da.Array:
    """
    Compute the analytic signal of `x` using the Hilbert transform.

    data: (n_times, n_signals)
    """
    time_axis = 0
    chunksize = data.chunksize[time_axis]
    chunk_overlap = chunksize // 4

    depth = dict(zip(range(data.ndim), [0] * data.ndim))
    depth[time_axis] = chunk_overlap  # Key: Axis index, Value: axis depth
    return da.map_overlap(
        npsig.hilbert, data, depth=depth, boundary="reflect", meta=data._meta
    )
