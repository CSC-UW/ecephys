from types import MappingProxyType
from typing import TYPE_CHECKING, Callable, Optional, Union

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import spikeinterface as si
    from spikeinterface.extractors.extractor_classes import KiloSortSortingExtractor

required_metric_thresholds = MappingProxyType(
    {
        "quality": {
            "all": {"good", "mua", np.nan, "noise"},
            "permissive": {"good", "mua", np.nan},
            "moderate": {"good", "mua", np.nan},
            "conservative": {"good", "mua", np.nan},
        },
        "firing_rate": {
            "all": (0.0, np.inf),
            "permissive": (0.2, np.inf),
            "moderate": (0.5, np.inf),
            "conservative": (0.5, np.inf),
        },
    }
)

isolation_metric_thresholds = MappingProxyType(
    {
        "isi_violations_ratio": {
            "permissive": (0.0, 0.5),
            "moderate": (0.0, 0.3),
            "conservative": (0.0, 0.1),
        },
        "rp_contamination": {
            "permissive": (0.0, 0.5),
            "moderate": (0.0, 0.3),
            "conservative": (0.0, 0.1),
        },
        "nn_isolation": {
            "permissive": (0.7, np.inf),
            "moderate": (0.8, np.inf),
            "conservative": (0.9, np.inf),
        },
    }
)

false_negative_metric_thresholds = MappingProxyType(
    {
        "amplitude_cutoff": {
            "permissive": (0.0, 0.499),
            "moderate": (0.0, 0.499),
            "conservative": (0.0, 0.3),
        }
    }
)

presence_metric_thresholds = MappingProxyType(
    {
        "presence_ratio": {
            "permissive": (0.8, np.inf),
            "moderate": (0.9, np.inf),
            "conservative": (0.9, np.inf),
        }
    }
)


# Type alias for sorting or dataframe - use string annotation to avoid runtime import
SortingOrDataFrame = Union["si.BaseSorting", pd.DataFrame]


def _check_sorting_or_dataframe(obj: SortingOrDataFrame) -> bool:
    import spikeinterface as si

    if not isinstance(obj, (si.BaseSorting, pd.DataFrame)):
        raise ValueError(
            f"Expected a SpikeInterface sorting or pandas DataFrame, got {type(obj)}"
        )
    return True


def _create_mask(obj: SortingOrDataFrame, fill_value: bool = True) -> np.ndarray:
    import spikeinterface as si

    _check_sorting_or_dataframe(obj)
    if isinstance(obj, si.BaseSorting):
        return np.full_like(obj.get_unit_ids(), fill_value)
    if isinstance(obj, pd.DataFrame):
        return np.full(len(obj), fill_value)


def _get_property(obj: SortingOrDataFrame, property: str) -> np.ndarray:
    import spikeinterface as si

    _check_sorting_or_dataframe(obj)
    if isinstance(obj, si.BaseSorting):
        return obj.get_property(property)
    if isinstance(obj, pd.DataFrame):
        return obj.get(property)


def _select_inviolate(
    obj: SortingOrDataFrame,
    thresholds: dict,
    threshold_level: str,
    metrics: list[str] = ["isi_violations_ratio", "rp_contamination"],
    nan: float = 0.0,
) -> np.ndarray:
    keep = _create_mask(obj, False)
    for m in metrics:
        v = np.array(_get_property(obj, m), dtype=float)
        v = np.nan_to_num(v, nan=nan)
        lo, hi = thresholds[m][threshold_level]
        passing = np.logical_and(v >= lo, v <= hi)
        keep = keep | passing
    return keep


def _select_present(
    obj: SortingOrDataFrame,
    thresholds: dict,
    threshold_level: str,
    nan: float = 1.0,
) -> np.ndarray:
    keep = _create_mask(obj, False)
    for m in ["presence_ratio_Wake", "presence_ratio_NREM", "presence_ratio_REM"]:
        v = np.array(_get_property(obj, m), dtype=float)
        v = np.nan_to_num(v, nan=nan)
        lo, hi = thresholds["presence_ratio"][threshold_level]
        passing = np.logical_and(v >= lo, v <= hi)
        keep = keep | passing
    return keep


def get_quality_metric_filters(
    required_threshold: str = "conservative",
    isolation_threshold: Optional[str] = "conservative",
    false_negatives_threshold: Optional[str] = "conservative",
    presence_threshold: Optional[str] = "conservative",
) -> tuple[dict, list[Callable]]:
    callable_filters = []
    simple_filters = {}

    required_simple_filters = {
        metric: required_metric_thresholds[metric][required_threshold]
        for metric in required_metric_thresholds
    }
    simple_filters.update(required_simple_filters)

    if isolation_threshold is not None:

        def select_inviolate(obj):
            return _select_inviolate(
                obj, isolation_metric_thresholds, isolation_threshold
            )

        callable_filters.append(select_inviolate)
        isolation_simple_filters = {
            metric: isolation_metric_thresholds[metric][isolation_threshold]
            for metric in ["nn_isolation"]
        }
        simple_filters.update(isolation_simple_filters)

    if false_negatives_threshold is not None:
        false_negative_simple_filters = {
            metric: false_negative_metric_thresholds[metric][false_negatives_threshold]
            for metric in false_negative_metric_thresholds
        }
        simple_filters.update(false_negative_simple_filters)

    if presence_threshold is not None:

        def select_present(obj):
            return _select_present(obj, presence_metric_thresholds, presence_threshold)

        callable_filters.append(select_present)

    return simple_filters, callable_filters


def add_anatomy_properties_to_extractor(
    extractor: "KiloSortSortingExtractor", structs: pd.DataFrame
) -> "KiloSortSortingExtractor":
    """
    Add a `structure` and `acronym` properties to each cluster indicating its anatomical region.

    Parameters
    ===========
    structure: The long structure name
    acronym: Abbreviated structure name
    hi: Upper boundary of the structure, in the same coordinates as the SI extractor's depth property
    lo: Lower boundary of the structure, in the same coordinates as the SI extractor's depth property

    Example structure table, as an HTSV file:
        structure	acronym	thickness	hi	lo
        Olfactory area / Basal forebrain / Dorsia tenia tecta	DTT	1192.4538258575196	1192.4538258575196	0.0
        Medial orbital cortex	MO	889.287598944591	2081.7414248021105	1192.4538258575196
        Prelimbic cortex / A32D + A32V	PreL	2506.174142480211	4587.915567282322	2081.7414248021105
        Secondary motor cortex	M2	2021.108179419525	6609.023746701847	4587.915567282322
        Out of brain	OOB	1050.9762532981529	7659.999999999999	6609.023746701847
    """

    depths = extractor.get_property("depth")
    structures = np.empty(depths.shape, dtype=object)
    acronyms = np.empty(depths.shape, dtype=object)
    for structure in structs.itertuples():
        lo = structure.lo
        hi = structure.hi
        mask = (depths >= lo) & (depths <= hi)
        structures[np.where(mask)] = structure.structure
        acronyms[np.where(mask)] = structure.acronym

    structures[pd.isnull(structures)] = "???"
    acronyms[pd.isnull(acronyms)] = "???"
    extractor.set_property("structure", structures)
    extractor.set_property("acronym", acronyms)
    extractor.set_annotation("structure_table", structs)
    return extractor


def get_dummy_structure_table(lo, hi):
    return pd.DataFrame(
        [{"structure": "Full probe", "acronym": "All", "lo": lo, "hi": hi}]
    )
