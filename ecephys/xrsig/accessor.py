import xarray as xr
from numbers import Number
from typing import Any, Hashable, Mapping

from xarray.core.utils import either_dict_or_kwargs


@xr.register_dataset_accessor("xrsig")
@xr.register_dataarray_accessor("xrsig")
class XRSigAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def is_coordinate_on_dimension(self, coord, dim):
        return coord in self._obj[dim].coords.keys()

    def get_dimension(self, coord):
        matching_dims = [
            dim for dim in self._obj.dims if self.is_coordinate_on_dimension(coord, dim)
        ]
        assert (
            len(matching_dims) == 1
        ), "Selection coordinate must be present on exatly one dimension."
        return matching_dims[0]

    def sel(
        self,
        indexers: Mapping[Hashable, Any] = None,
        method: str = None,
        tolerance: Number = None,
        drop: bool = False,
        **indexers_kwargs: Any,
    ):
        indexers = either_dict_or_kwargs(indexers, indexers_kwargs, "sel")
        indexer_coords = [each for each in indexers if not each in self._obj.dims]
        dims_to_swap = {self.get_dimension(coord): coord for coord in indexer_coords}
        return (
            self._obj.swap_dims(dims_to_swap)
            .sel(
                indexers=indexers,
                drop=drop,
                method=method,
                tolerance=tolerance,
            )
            .swap_dims({value: key for key, value in dims_to_swap.items()})
        )