import dask.array as da
import numpy as np
import yasa

from ecephys import npsig


def shift_blocks(
    data: da.Array, shifts: np.ndarray, axis: int, **take_kwargs
) -> da.Array:
    """Shifts has the same shape as data.blocks. Axis is the axis along which each block is shifted.
    The returned dask array has the same shape, dtype, and chunks as data."""

    def _shift_block(block: np.ndarray, block_id: tuple = None) -> np.ndarray:
        indices = np.arange(block.shape[axis])
        return npsig.take(block, indices + shifts[block_id], axis=axis, **take_kwargs)

    return da.map_blocks(_shift_block, data, meta=data._meta)


def moving_transform(
    data: da.Array, fs: float, window: float, step: float, method: str
) -> da.Array:
    assert data.ndim == 2, "Data must be 2D."
    time_axis = 0
    channel_axis = 1

    padlen = int(window * fs)
    chunk_overlap = np.round(2 * padlen).astype(int)
    min_chunksize = 3 * chunk_overlap
    if data.chunksize[time_axis] < min_chunksize:
        raise ValueError(
            f"Chunks are too small for window size. Chunks should be at least {min_chunksize} samples at {fs} Hz. Please rechunk."
        )

    depth = dict(zip(range(data.ndim), [0] * data.ndim))
    depth[time_axis] = chunk_overlap  # Key: Axis index, Value: axis depth

    def _moving_transform(x: np.ndarray) -> np.ndarray:
        mrms = np.zeros_like(x)
        for i in range(x.shape[channel_axis]):
            _, mrms[:, i] = yasa.moving_transform(
                x=x[:, i], sf=fs, window=window, step=step, method=method, interp=True
            )
        return mrms

    transformed = da.map_overlap(
        _moving_transform, data, depth=depth, boundary="reflect", meta=data._meta
    )
    return transformed
