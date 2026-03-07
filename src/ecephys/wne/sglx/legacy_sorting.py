# TODO: These functions should probably be moved to the `legacy_npix_sorting_pipeline`
# package.
# TODO: Functions that load data from disk should rename any column with "segment" in
# its name to "slice", to avoid confusion with SpikeInterface segments.
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np
import pandas as pd

import ecephys.utils
from ecephys.wne import constants
from ecephys.wne.sglx.project import SGLXProject

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# Was `SGLXProject.get_si_recording()`
# TODO: Consider making this a method on wne.Project, or at least a function in wne.sorting
# Although, I don't really like the idea that it uses the alias. But the existing data could be moved to eliminate the alias.
def get_sorting_directory(
    project: SGLXProject,
    subject: str,
    experiment: str,
    alias: str,
    probe: str,
    sorting: str,
) -> Path:
    return (
        project.get_alias_subject_directory(experiment, alias, subject)
        / f"{sorting}.{probe}"
    )


def get_sorting_file(
    project: SGLXProject,
    subject: str,
    experiment: str,
    alias: str,
    probe: str,
    sorting: str,
    file_name: str,
) -> Path:
    return (
        get_sorting_directory(project, subject, experiment, alias, probe, sorting)
        / file_name
    )


def load_slice_table_from_sorting_folder(
    project: SGLXProject,
    subject: str,
    experiment: str,
    alias: str,
    probe: str,
    sorting: str,
    return_excised_slices: bool = False,  # If False, only return slices of type "keep"
) -> pd.DataFrame:
    """Load a sorting's slice table from disk.

    Add a couple useful columns: `segmentExpmtPrbAcqFirstTime`, `segmentExpmtPrbAcqLastTime`
    """  # TODO: Useful for what?
    slice_table_file = get_sorting_file(
        project, subject, experiment, alias, probe, sorting, "segments.htsv"
    )
    if not slice_table_file.exists():
        raise FileNotFoundError(f"Slice table not found at {slice_table_file}.")

    slice_table = ecephys.utils.read_htsv(slice_table_file)
    slice_table = slice_table.rename(
        columns={"nSegmentSamp": "n_slice_samples", "type": "sliceType"}
    )

    slice_table["n_slice_samples"] = (
        slice_table["withinFileEndFrame"] - slice_table["withinFileStartFrame"]
    )  # TODO: This column should already be present. Also, what is it used for? Document in a schema.
    slice_table["segmentDuration"] = (
        slice_table["n_slice_samples"].astype(float).div(slice_table["imSampRate"])
    )  # TODO: This column should already be present. Also, what is it used for? Document in a schema.
    slice_table["segmentExpmtPrbAcqFirstTime"] = slice_table[
        "expmtPrbAcqFirstTime"
    ] + slice_table["withinFileStartFrame"].astype(float).div(slice_table["imSampRate"])
    # For LastTime, we work backwards from expmtPrbAcqLastTime (rather than forward
    # from expmtPrbAcqFirstTime) to ensure that segmentExpmtPrbAcqLastTime <= expmtPrbAcqLastTime
    # This is not the case when working forward due to floating point errors
    # segments["segmentExpmtPrbAcqLastTime"] = (
    #     segments["segmentExpmtPrbAcqFirstTime"] + segments["segmentDuration"]
    # ) # Nope
    slice_table["segmentExpmtPrbAcqLastTime"] = slice_table["expmtPrbAcqLastTime"] - (
        slice_table["nFileSamp"] - slice_table["withinFileEndFrame"]
    ).astype(float).div(slice_table["imSampRate"])

    assert np.all(
        slice_table["segmentExpmtPrbAcqFirstTime"]
        >= slice_table["expmtPrbAcqFirstTime"]
    )
    assert np.all(
        slice_table["segmentExpmtPrbAcqLastTime"] <= slice_table["expmtPrbAcqLastTime"]
    )

    if return_excised_slices:
        return slice_table

    return slice_table[slice_table["sliceType"] == "keep"]


# This requires a segment table to have been created and saved to disk.
# It is therefore not general, and is only intended to be used for sorting results.
# TODO: Create get_times_from_sorting() with wne.sglx.utils.slice_table2times().
def get_sample2time_from_sorting(
    project: SGLXProject,
    subject: str,
    experiment: str,
    alias: str,
    probe: str,
    sorting: str,
    allow_no_sync_file: bool = False,
) -> Callable:
    from ecephys.wne.sglx import utils as sglx_utils

    slice_table = load_slice_table_from_sorting_folder(
        project,
        subject,
        experiment,
        alias,
        probe,
        sorting,
        return_excised_slices=False,
    ).copy()

    # Get sync table
    sync_file = project.get_experiment_subject_file(
        experiment, subject, constants.Files.AP_SYNC
    )
    if not sync_file.exists():
        print(f"Sync table not found at {sync_file}")
        if allow_no_sync_file:
            print("`allow_no_sync_file` == True : Ignoring probe sync")
            sync_table = None
        else:
            raise FileNotFoundError(f"No sync file at {sync_file}")
    else:
        sync_table = ecephys.utils.read_htsv(sync_file)

    slice_table = sglx_utils.add_sample2time_columns(slice_table, sync_table)
    return sglx_utils.get_sample2time(slice_table)


def load_singleprobe_sorting(
    sglx_sorting_project: SGLXProject,
    subject: str,
    experiment: str,
    probe: str,
    alias: str = "full",
    sorting: str = "sorting",
    postprocessing: str = "postpro",
    wne_anatomy_project: Optional[SGLXProject] = None,
    allow_no_sync_file=False,
) -> "ecephys.units.siks_sorting.SpikeInterfaceKilosortSorting":
    from ecephys.units.siks_sorting import SpikeInterfaceKilosortSorting
    from ecephys.wne import siutils

    if sorting is None:
        sorting = "sorting"
    if postprocessing is None:
        postprocessing = "postpro"

    # Get function for converting SI samples to imec0 timebase
    sample2time = get_sample2time_from_sorting(
        sglx_sorting_project,
        subject,
        experiment,
        alias,
        probe,
        sorting,
        allow_no_sync_file=allow_no_sync_file,
    )

    # Load extractor
    extractor = sglx_sorting_project.get_kilosort_extractor(
        subject,
        experiment,
        probe,
        alias=alias,
        sorting=sorting,
        postprocessing=postprocessing,
    )

    # TODO: Why was this removed, and should it be restored?
    # extractor = units.si_ks_sorting.fix_isi_violations_ratio(extractor)

    # Add anatomy to the extractor, if available.
    if wne_anatomy_project is None:
        wne_anatomy_project = sglx_sorting_project

    anatomy_file = wne_anatomy_project.get_experiment_subject_file(
        experiment, subject, f"{probe}.structures.htsv"
    )
    if anatomy_file.exists():
        structs = ecephys.utils.read_htsv(anatomy_file)
    else:
        import warnings

        warnings.warn(
            "Could not find anatomy file at: {anatomy_file}. Using dummy structure table"
        )
        structs = siutils.get_dummy_structure_table(lo=-np.inf, hi=np.inf)
    extractor = siutils.add_anatomy_properties_to_extractor(extractor, structs)

    return SpikeInterfaceKilosortSorting(extractor, sample2time)


def load_multiprobe_sorting(
    sglx_sorting_project: SGLXProject,
    subject: str,
    experiment: str,
    probes: list[str],
    alias: str = "full",
    sortings: dict[str, str] = None,
    postprocessings: dict[str, str] = None,
    wne_anatomy_project: Optional[SGLXProject] = None,
    allow_no_sync_file=False,
) -> "ecephys.units.multi_siks.MultiSIKS":
    from ecephys.units.multi_siks import MultiSIKS

    if sortings is None:
        sortings = {prb: None for prb in probes}
    if postprocessings is None:
        postprocessings = {prb: None for prb in probes}

    return MultiSIKS(
        {
            probe: load_singleprobe_sorting(
                sglx_sorting_project,
                subject,
                experiment,
                probe=probe,
                alias=alias,
                sorting=sortings[probe],
                postprocessing=postprocessings[probe],
                wne_anatomy_project=wne_anatomy_project,
                allow_no_sync_file=allow_no_sync_file,
            )
            for probe in probes
        }
    )
