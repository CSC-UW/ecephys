import dask.array
import numpy as np

from ecephys import npsig


def shift_blocks(
    data: dask.array.Array, shifts: np.ndarray, axis: int, **take_kwargs
) -> dask.array.Array:
    """Shifts has the same shape as data.blocks. Axis is the axis along which each block is shifted.
    The returned dask array has the same shape, dtype, and chunks as data."""

    def _shift_block(block: np.ndarray, block_id: tuple = None) -> np.ndarray:
        indices = np.arange(block.shape[axis])
        return npsig.take(block, indices + shifts[block_id], axis=axis, **take_kwargs)

    return dask.array.map_blocks(_shift_block, data, meta=data._meta)
