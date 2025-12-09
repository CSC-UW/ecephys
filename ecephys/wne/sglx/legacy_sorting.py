# TODO: These functions should probably be moved to the `legacy_npix_sorting_pipeline`
# package.
# TODO: Functions that load data from disk should rename any column with "segment" in
# its name to "slice", to avoid confusion with SpikeInterface segments.
import logging
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import spikeinterface as si
import spikeinterface.extractors as se

import ecephys.utils
from ecephys import units, wne
from ecephys.wne.sglx.project import SGLXProject
from ecephys.wne.sglx.subject import SGLXSubject

logger = logging.getLogger(__name__)


def _create_slice_table_for_spikeinterface(
    subject_ftab: pd.DataFrame, exclusions: pd.DataFrame
) -> pd.DataFrame:
    df = wne.sglx.utils.create_slice_table_for_spikeinterface(subject_ftab, exclusions)

    # TODO: Remove, as these can (1) be derived later, and (2) get overwritten in
    #       load_slice_table_from_sorting_folder().
    df["nSegmentSamp"] = df["withinFileEndFrame"] - df["withinFileStartFrame"]
    df["segmentDuration"] = df["nSegmentSamp"].div(df["imSampRate"])

    return df


# Was `SGLXProject.get_si_recording()`
# TODO: I would make this a private function, to discourage its use.
def get_recording(
    subject: SGLXSubject,
    experiment: str,
    alias: str,
    stream: str,
    probe: str,
    combine: str = "concatenate",
    exclusions: Optional[pd.DataFrame] = None,
    sampling_frequency_max_diff: Optional[float] = 1e-6,
) -> tuple[si.BaseRecording, pd.DataFrame]:
    """Combine the one or more recordings comprising an experiment or alias into a single SI recording object.

    Parameters
    ==========
    combine: 'concatenate' or 'append'
        If 'concatenate' (default), the returned recording object is one single monolothic segment.
        This is the default behavior, because SI sorters currently only work on single-segment recordings.
        If 'append', the returned recording object consists of multiple segments.
        This might be useful for certain preprocessing, postprocessing operations, etc.
    exclusions:
        Specify which parts of the recording to drop. We slice such that the first and last samples
        of each exclusion are NOT included in the returned recording.
        fname: The name of the file (e.g. 3-2-2021_J_g0_t1.imec1.ap.bin)
        withinFileStartTime: The start time of the data to drop, in seconds from the start of the file.
        withinFileEndTime: The end time of the data to drop, in seconds from the start of the file.
            If greater than the file duration, the excess time will be ignored, not dropped from the next file.
        type: A label you can assign to keep track of why this data was excluded. As long as the value is not "keep", the data will be dropped.

    Returns
    =======
    recording:
        The combined SI recording object.
    exclusions:
        A dataframe where each row is a slice of data to keep, or drop, sorted in chronological order.
            fname: The name of the file (e.g. 3-2-2021_J_g0_t1.imec1.ap.bin)
            withinFileStartFrame: The first sample index of the slice, measured from the start of the file (0-indexed)
            withinFileEndFrame: The final sample index of the slice, measured from the start of the file (0-indexed)
            type: Either 'keep', in which case the slice was kept, or other, in which case the slice was dropped.
            segmentDuration: Duration in sec of slice.
    """
    # TODO: Instead of anticipating SI segment indices and adding them to the ftab at the start,
    # we should use the neo header in the SI extractor to add the segment indices to an ftab,
    # or to our slice table.
    #
    # Get the experiment frame. This should be for a single probe, and a single stream.
    ftab = subject.get_experiment_frame(
        experiment, alias=alias, stream=stream, ftype="bin", probe=probe
    )
    # Split the experiment frame around the exclusions, using precise sample indices.
    if exclusions is None:
        exclusions = wne.utils.get_dummy_artifacts_table()
    slices = _create_slice_table_for_spikeinterface(
        ftab, exclusions
    )  # These are NOT the segments of a SpikeInterface recording!

    # Take the good slices one by one, create an recording object for each, and save these all in a list
    good_slices = slices[slices["type"] == "keep"]
    recordings = list()
    for slice_ in good_slices.itertuples():
        extractor = se.SpikeGLXRecordingExtractor(
            slice_.gate_dir.parent, stream_id=f"{probe}.{stream}"
        )  # slice_.gate_dir.parent is the actual gate directory.
        recording = extractor.select_segments(
            [slice_.gate_dir_trigger_file_idx]
        ).frame_slice(
            start_frame=slice_.withinFileStartFrame,
            end_frame=slice_.withinFileEndFrame,
        )
        recordings.append(recording)

    if combine == "concatenate":
        recording = si.concatenate_recordings(
            recordings, sampling_frequency_max_diff=sampling_frequency_max_diff
        )
    elif combine == "append":
        recording = si.append_recordings(
            recordings, sampling_frequency_max_diff=sampling_frequency_max_diff
        )
    else:
        raise ValueError(f"Got unexpected value for `combine`: {combine}")

    # We return both recording and slices together, rather than making the available separately,
    # to ensure that you never get a slice table unless it is actually proven to produce a valid extractor object.
    return recording, slices


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

    slice_table["nSegmentSamp"] = (
        slice_table["withinFileEndFrame"] - slice_table["withinFileStartFrame"]
    )  # TODO: This column should already be present. Also, what is it used for? Document in a schema.
    slice_table["segmentDuration"] = (
        slice_table["nSegmentSamp"].astype(float).div(slice_table["imSampRate"])
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

    return slice_table[slice_table["type"] == "keep"]


# This requires a segment table to have been created and saved to disk.
# It is therefore not general, and is only intended to be used for sorting results.
def get_sample2time_from_sorting(
    project: SGLXProject,
    subject: str,
    experiment: str,
    alias: str,
    probe: str,
    sorting: str,
    allow_no_sync_file: bool = False,
) -> Callable:
    slice_table = load_slice_table_from_sorting_folder(
        project,
        subject,
        experiment,
        alias,
        probe,
        sorting,
        return_excised_slices=False,
    ).copy()
    return wne.sglx.utils.get_sample2time(
        project, subject, experiment, slice_table, allow_no_sync_file
    )


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
) -> units.SpikeInterfaceKilosortSorting:
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
        structs = wne.siutils.get_dummy_structure_table(lo=-np.inf, hi=np.inf)
    extractor = wne.siutils.add_anatomy_properties_to_extractor(extractor, structs)

    return units.SpikeInterfaceKilosortSorting(extractor, sample2time)


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
) -> units.MultiSIKS:
    if sortings is None:
        sortings = {prb: None for prb in probes}
    if postprocessings is None:
        postprocessings = {prb: None for prb in probes}

    return units.MultiSIKS(
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
