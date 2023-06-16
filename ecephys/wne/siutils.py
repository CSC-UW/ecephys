from typing import Callable, Optional

import numpy as np
import spikeinterface as si

required_metric_thresholds = {
    "quality": {
        "permissive": {"good", "mua", np.NaN},
        "moderate": {"good", "mua", np.NaN},
        "conservative": {"good", "mua", np.NaN},
    },
    "firing_rate": {
        "permissive": (0.2, np.Inf),
        "moderate": (0.5, np.Inf),
        "conservative": (0.5, np.Inf),
    },
}

isolation_metric_thresholds = {
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
        "permissive": (0.7, np.Inf),
        "moderate": (0.8, np.Inf),
        "conservative": (0.9, np.Inf),
    },
}

false_negative_metric_thresholds = {
    "amplitude_cutoff": {
        "permissive": (0.0, 0.499),
        "moderate": (0.0, 0.499),
        "conservative": (0.0, 0.3),
    }
}

presence_metric_thresholds = {
    "presence_ratio": {
        "permissive": (0.8, np.Inf),
        "moderate": (0.9, np.Inf),
        "conservative": (0.9, np.Inf),
    }
}


def _select_inviolate(
    si_obj: si.BaseSorting,
    thresholds: dict,
    threshold_level: str,
    metrics: list[str] = ["isi_violations_ratio", "rp_contamination"],
    nan: float = 0.0,
) -> np.ndarray[bool]:
    keep = np.zeros_like(si_obj.get_unit_ids())
    for m in metrics:
        v = si_obj.get_property(m)
        v = np.nan_to_num(v, nan)
        lo, hi = thresholds[m][threshold_level]
        passing = np.logical_and(v >= lo, v <= hi)
        keep = keep | passing
    return keep


def _select_present(
    si_obj: si.BaseSorting, thresholds: dict, threshold_level: str, nan: float = 1.0
) -> np.ndarray[bool]:
    keep = np.zeros_like(si_obj.get_unit_ids())
    for m in ["presence_ratio_Wake", "presence_ratio_NREM", "presence_ratio_REM"]:
        v = si_obj.get_property(m)
        v = np.nan_to_num(v, nan)
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

        def select_inviolate(si_obj):
            return _select_inviolate(
                si_obj, isolation_metric_thresholds, isolation_threshold
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

        def select_present(si_obj):
            return _select_present(
                si_obj, presence_metric_thresholds, presence_threshold
            )

        callable_filters.append(select_present)

    return simple_filters, callable_filters
