import logging
from typing import Callable, Optional

import numpy as np
import pandas as pd
import spikeinterface.extractors as se
from pandas.api import types

logger = logging.getLogger(__name__)


def refine_clusters(
    si_obj: se.BaseSorting,
    simple_filters: Optional[dict] = None,
    callable_filters: Optional[list[Callable]] = None,
    include_nans: bool = True,
    verbose: bool = True,
):
    """Subselect clusters based on filters.

    Parameters:
    ===========
    si_obj: SI extractor or sorting object
    filters: dict
        Keys are the names of cluster properties.  Example properties include all columns in cluster_info.tsv, including quality metrics.
        Values are either 2-item tuples or sets.
        Tuples specify a range of allowable values for non-categorical numeric properties.
        Sets specify allowable values for categorical properties.
        For example, {"n_spikes": (2, np.inf)} will load only clusters with 2 or more spikes.
        For example, {"quality": {"good", "mua"}} will load only clusters marked as such after curation and QMs.
    include_nans: bool, default True
        For several properties/metrics (including the cluster "group"/"quality" possibly set during curation),
        the value of the property may be np.nan for some clusters. If True, we include those clusters (effectively
        filtering based on a property only when this property has a valid value)


    Returns:
    ========
    UnitsSelectionSorting

    Notes:
    ======
    SpikeInterface renames the 'group' columns in cluster_info.tsv to 'quality'.
    """
    keep = np.ones_like(si_obj.get_unit_ids())
    if simple_filters is not None:
        for property, filter in simple_filters.items():
            values = si_obj.get_property(property)
            if property not in si_obj.get_property_keys():
                logger.warning(
                    f"Cluster property {property} not found. Unable to filter clusters based on {property}."
                )
                continue
            if isinstance(filter, tuple):
                lo, hi = filter
                assert types.is_numeric_dtype(np.array(filter)), (
                    f"Expected a numeric dtype for values in tuple: `{filter}`. Specify values of interest as a set (rather than tuple) for non-numerical properties."
                )
                assert types.is_numeric_dtype(values.dtype), (
                    f"Cannot select a range of values for cluster property {property} with dtype {values.dtype}. Expected a numeric dtype."
                )
                mask = np.logical_and(values >= lo, values <= hi)
            elif isinstance(filter, set):
                mask = np.isin(values, list(filter))
            else:
                raise ValueError(
                    f"Cluster property {property} was provided as type {type(filter)}. Expected a tuple for selecting a range of numerical values, or a set for selecting categorical variables."
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
            mask = filter_func(si_obj)
            keep = keep & mask
            if verbose:
                print(
                    f"Callable filter {filter_func.__name__} excludes {mask.size - mask.sum()} clusters."
                )

    if verbose:
        print(
            f"{keep.size - keep.sum()}/{keep.size} clusters excluded by jointly applying filters. {keep.sum()} remain."
        )
    clusterIDs = si_obj.get_unit_ids()[np.where(keep)]
    return si_obj.select_units(clusterIDs)
