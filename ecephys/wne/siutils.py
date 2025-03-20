from types import MappingProxyType
from typing import Callable, Optional

import numpy as np
import pandas as pd
import spikeinterface as si
import spikeinterface.extractors as se
from spikeinterface.core import waveform_tools

import ecephys.utils

from .project import Project
from .subject import Subject

required_metric_thresholds = MappingProxyType(
    {
        "quality": {
            "permissive": {"good", "mua", np.nan},
            "moderate": {"good", "mua", np.nan},
            "conservative": {"good", "mua", np.nan},
        },
        "firing_rate": {
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


def load_postprocessing_hypnogram_for_slicing(
    sorting_project: Project,
    subject: Subject,
    experiment: str,
    probe: str,
    alias: str = "full",
    sorting: str = "sorting",
    postprocessing: str = "postpro",
    drop_time_columns: bool = True,
) -> pd.DataFrame:
    """Load postprocessing hypnogram, which can be used with si.frame_slice

    Important:
    This is NOT adequate for use as regular hypnogram since the
    start/end_time and duration fields do not account for gaps!
    But the start_sample,end_sample columns can be used with
    the si.frame_slice() methods.
    However, this may be used as regular hypnogram after reconciliating with
    exclusions.
    """
    f = (
        sorting_project.get_alias_subject_directory(experiment, alias, subject.name)
        / f"{sorting}.{probe}"
        / postprocessing
        / "hypnogram.htsv"
    )

    if not f.exists():
        import warnings

        warnings.warn("No `hypnogram.htsv` file in postpro dir. Returning None")
        return None

    df = ecephys.utils.read_htsv(f)
    if drop_time_columns:
        # Drop misleading start/end_time/duration columns
        return df.drop(columns=["start_time", "end_time", "duration"])

    return df


# TODO: Remove unused combine parameter,
# and maybe rename to slice_extractor_and_concatenate_segments.
# TODO: This is very slow. ~15m. Why?
def cut_and_combine_si_extractors(si_object, epochs_df, combine="concatenate"):
    """Slices a single extractor (that probably represents a whole recording)
    into pieces (e.g., artifact-free epochs, or epochs belonging to a
    condition of interest), and recombines them together."""
    assert {"start_frame", "end_frame", "state"}.issubset(epochs_df)
    assert len(epochs_df.state.unique()) == 1

    if not isinstance(si_object, (se.BaseSorting, se.BaseRecording)):
        raise ValueError(
            "Unrecognized datatype for si_object. "
            "Expected spikeinterface BaseSorting or BaseRecording."
        )

    frame_slice_kwargs = {}
    if isinstance(si_object, se.BaseSorting):
        # Disable redundant check_spike_frames in Sorting.frame_slice
        assert si_object.has_recording()
        if waveform_tools.has_exceeding_spikes(si_object._recording, si_object):
            raise ValueError(
                "The sorting object has spikes exceeding the recording duration. You have to remove those spikes "
                "with the `spikeinterface.curation.remove_excess_spikes()` function"
            )
        frame_slice_kwargs = {"check_spike_frames": False}

    si_segments = []
    for epoch in epochs_df.itertuples():
        si_segments.append(
            si_object.frame_slice(
                start_frame=epoch.start_frame,
                end_frame=epoch.end_frame,
                **frame_slice_kwargs,
            )
        )

    if combine == "concatenate":
        if isinstance(si_object, se.BaseSorting):
            rec = si.concatenate_sortings(si_segments)
        elif isinstance(si_object, se.BaseRecording):
            rec = si.concatenate_recordings(si_segments)

    elif combine == "append":
        raise NotImplementedError

    else:
        assert False

    # Apply to time vector if there's any (not handled by SI)
    # TODO: "not handled by SI": I wouldn't be so sure. A lot has changed.
    if si_object.has_time_vector():
        raw_times = si_object.get_times()
        times = []
        for epoch in epochs_df.itertuples():
            times += list(raw_times[epoch.start_frame : epoch.end_frame])
        times = np.array(times)
        rec.set_times(times, with_warning=False)

    return rec


def add_anatomy_properties_to_extractor(
    extractor: se.KiloSortSortingExtractor, structs: pd.DataFrame
) -> se.KiloSortSortingExtractor:
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
