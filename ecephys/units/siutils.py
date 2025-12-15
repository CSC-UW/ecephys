import logging
from typing import Callable, Optional

import numpy as np
import pandas as pd
import spikeinterface.full as si
from pandas.api import types

logger = logging.getLogger(__name__)


def _check_sorting_or_dataframe(obj: si.BaseSorting | pd.DataFrame) -> bool:
    if not isinstance(obj, (si.BaseSorting, pd.DataFrame)):
        raise ValueError(
            f"Expected a SpikeInterface sorting or pandas DataFrame, got {type(obj)}"
        )
    return True


def _create_mask(
    obj: si.BaseSorting | pd.DataFrame, fill_value: bool = True
) -> np.ndarray[bool]:
    _check_sorting_or_dataframe(obj)
    if isinstance(obj, si.BaseSorting):
        return np.full_like(obj.get_unit_ids(), fill_value)
    if isinstance(obj, pd.DataFrame):
        return np.full(len(obj), fill_value)


def _get_property(obj: si.BaseSorting | pd.DataFrame, property: str) -> np.ndarray:
    _check_sorting_or_dataframe(obj)
    if isinstance(obj, si.BaseSorting):
        return obj.get_property(property)
    if isinstance(obj, pd.DataFrame):
        return obj.get(property)


def _has_property(obj: si.BaseSorting | pd.DataFrame, property: str) -> bool:
    _check_sorting_or_dataframe(obj)
    if isinstance(obj, si.BaseSorting):
        return property in obj.get_property_keys()
    if isinstance(obj, pd.DataFrame):
        return property in obj.columns


def _select(
    obj: si.BaseSorting | pd.DataFrame, mask: np.ndarray[bool]
) -> si.UnitsSelectionSorting | pd.DataFrame:
    _check_sorting_or_dataframe(obj)
    if isinstance(obj, si.BaseSorting):
        cluster_ids = obj.get_unit_ids()[np.where(mask)]
        return obj.select_units(cluster_ids)
    if isinstance(obj, pd.DataFrame):
        return obj.loc[mask]


def refine_clusters(
    obj: si.BaseSorting | pd.DataFrame,
    simple_filters: Optional[dict] = None,
    callable_filters: Optional[list[Callable]] = None,
    include_nans: bool = True,
    verbose: bool = True,
) -> si.UnitsSelectionSorting | pd.DataFrame:
    """Subselect clusters based on filters.

    Parameters:
    ===========
    obj: SI extractor or sorting object, or pandas DataFrame
    simple_filters: dict, optional
    simple_filters: dict, optional
        Keys are the names of cluster properties.
        Example properties include all columns in Kilosort's cluster_info.tsv,
        including quality metrics.
        Values are either 2-item tuples or sets.
        - Tuples specify a range of allowable values for non-categorical numeric properties.
        - Sets specify allowable values for categorical properties.
        For example:
        - {"n_spikes": (2, np.inf)} will load only clusters with 2 or more spikes.
        - {"quality": {"good", "mua"}} will load only clusters marked as such after curation and QMs.
    callable_filters: list[Callable], optional
        List of functions that can take either a sorting object OR a pandas DataFrame
        and return a boolean mask of the same length as the input.
    include_nans: bool, default True
        For several properties/metrics (including the cluster "group"/"quality" possibly
        set during curation), the value of the property may be np.nan for some clusters.
        If True, we include those clusters (effectively filtering based on a property
        only when this property has a valid value)

    Returns:
    ========
    si.BaseSorting | pd.DataFrame

    Notes:
    ======
    SpikeInterface renames the 'group' columns in cluster_info.tsv to 'quality'.
    """
    keep = _create_mask(obj)
    if simple_filters is not None:
        for property, filter in simple_filters.items():
            values = _get_property(obj, property)
            if not _has_property(obj, property):
                logger.warning(
                    f"Cluster property {property} not found. "
                    f"Unable to filter clusters based on {property}."
                )
                continue
            if isinstance(filter, tuple):
                lo, hi = filter
                if not types.is_numeric_dtype(np.array(filter)):
                    raise ValueError(
                        f"Expected a numeric dtype for values in tuple: `{filter}`. "
                        f"Specify values of interest as a set (rather than tuple) for "
                        f"non-numerical properties."
                    )
                if not types.is_numeric_dtype(values):
                    raise ValueError(
                        f"Cannot select a range of values for cluster property "
                        f"{property} with dtype {values.dtype}. "
                        f"Expected a numeric dtype."
                    )
                mask = np.logical_and(values >= lo, values <= hi)
            elif isinstance(filter, set):
                mask = np.isin(values, list(filter))
            else:
                raise ValueError(
                    f"Cluster property {property} was provided as type {type(filter)}. "
                    f"Expected a tuple for selecting a range of numerical values, "
                    f"or a set for selecting categorical variables."
                )
            if include_nans:
                mask = pd.isna(values) | mask
            keep = keep & mask
            if verbose:
                print(
                    f"{property}: {filter} excludes {mask.size - mask.sum()} clusters."
                )

    if callable_filters is not None:
        for filter_func in callable_filters:
            mask = filter_func(obj)
            keep = keep & mask
            if verbose:
                print(
                    f"Callable filter {filter_func.__name__} excludes "
                    f"{mask.size - mask.sum()} clusters."
                )

    if verbose:
        print(
            f"{keep.size - keep.sum()}/{keep.size} clusters excluded by jointly "
            f"applying filters. {keep.sum()} remain."
        )
    return _select(obj, keep)
