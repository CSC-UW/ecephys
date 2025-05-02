import itertools
import logging
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import spikeinterface as si
import spikeinterface.extractors as se
import tqdm

import ecephys.utils
from ecephys import hypnogram as hyp
from ecephys import units, wne
from ecephys.wne.sglx import utils
from ecephys.wne.sglx.project import SGLXProject
from ecephys.wne.sglx.subject import SGLXSubject

logger = logging.getLogger(__name__)

MIN_BOUT_DURATION_SEC = 0.1  # SUS: Why is this a module level constant? Why is it not just a pre-defined parameter? Why not a WNE constant?


# TODO: This should be renamed to _split_experiment_frame_for_spikeinterface() to avoid confusion with the `segments` property of a SpikeInterface recording.
def _segment_experiment_frame_for_spikeinterface(
    subject_ftab: pd.DataFrame, exclusions: pd.DataFrame
) -> pd.DataFrame:
    """Split an experiment frame for a single subject, probe, stream, and filetype around a set of periods to exclude.
    These segments do NOT correspond to the `segments` property of a SpikeInterface recording. Sorry.
    For details, see `get_si_recording()`.
    """
    EXCLUSION_COLS = ["withinFileStartTime", "withinFileEndTime", "fname"]
    assert all([c in exclusions.columns for c in EXCLUSION_COLS]), (
        f"Invalid columns for exclusions. Expected: `{EXCLUSION_COLS}`"
    )
    segments = list()
    # For each file in the experiment, split it if necessary.
    # If not, just create a segment that is the entire file.
    for file in subject_ftab.itertuples():
        ns = file.nFileSamp
        fname = file.path.name
        mask = (
            exclusions["fname"] == fname
        )  # Get the exclusions pertaining to this file.

        # For the exclusions pertaining to this file, convert their definition in seconds to precise sample indices,
        # and clip these estimates so that sample indices don't extend beyond the ends of the file.
        exclusions.loc[mask, "withinFileStartFrame"] = (
            (exclusions.loc[mask, "withinFileStartTime"] * file.imSampRate)
            .astype(int)
            .clip(0, ns)
        )
        exclusions.loc[mask, "withinFileEndFrame"] = (
            (exclusions.loc[mask, "withinFileEndTime"] * file.imSampRate)
            .astype(int)
            .clip(0, ns)
        )

        # Do the actual splitting of the entire file around the exclusions
        file_segments = ecephys.utils.pandas.reconcile_labeled_intervals(
            exclusions.loc[
                mask, ["withinFileStartFrame", "withinFileEndFrame", "type"]
            ],
            pd.DataFrame(
                {
                    "withinFileStartFrame": [0],
                    "withinFileEndFrame": [ns],
                    "type": "keep",
                }
            ),
            "withinFileStartFrame",
            "withinFileEndFrame",
        ).drop(columns="delta")
        file_segments["fname"] = fname

        # The function above considers intervals to be open-ended, so that (a, b) and (b, c) are considered NON-overlapping
        # (as for usual python slicing)
        # This means that up to this point the end sample of an exclusion will be part of the next (kept) segment.
        # In order to be conservative, we correct each bad segment followed by a good segment to include its last sample.
        # For example, if (a, b) and (b, c) are bad segment, and (c, d) is a good segment, the new segments will be (a, b+1), (b+1, c), (c+1, d).
        keep = file_segments["type"] == "keep"
        frames_to_shift = np.intersect1d(
            file_segments[~keep]["withinFileEndFrame"].values,
            file_segments["withinFileStartFrame"].values,
        )  # End of each bad segment followed by another segment (excludes the last one)
        i = file_segments["withinFileEndFrame"].isin(frames_to_shift)
        j = file_segments["withinFileStartFrame"].isin(frames_to_shift)
        file_segments.loc[i, "withinFileEndFrame"] += 1
        file_segments.loc[j, "withinFileStartFrame"] += 1

        # Do some sanity checks, ensuring that every sample in the file is accounted for.
        assert file_segments["withinFileStartFrame"].min() == 0, (
            "Something went wrong when splitting file around exclusions."
        )
        assert file_segments["withinFileEndFrame"].max() == (ns), (
            "Something went wrong when splitting file around exclusions."
        )
        segments.append(file_segments)
        assert (
            file_segments["withinFileEndFrame"] - file_segments["withinFileStartFrame"]
        ).sum() == ns, "Something went wrong when splitting file around exclusions."

    # Return the segments, adding metadata about the files that they come from, for convenience.
    segments = pd.concat(segments, ignore_index=True).astype(
        {"withinFileStartFrame": int, "withinFileEndFrame": int}
    )
    subject_ftab["fname"] = subject_ftab["path"].apply(lambda x: x.name)

    stab = segments.merge(subject_ftab, on="fname")

    stab["nSegmentSamp"] = stab["withinFileEndFrame"] - stab["withinFileStartFrame"]
    stab["segmentDuration"] = stab["nSegmentSamp"].div(stab["imSampRate"])

    return stab


# Was `SGLXProject.get_si_recording()`
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
    segments:
        A dataframe where each row is a segment of data to keep, or drop, sorted in chronological order.
            fname: The name of the file (e.g. 3-2-2021_J_g0_t1.imec1.ap.bin)
            withinFileStartFrame: The first sample index of the segment, measured from the start of the file (0-indexed)
            withinFileEndFrame: The final sample index of the segment, measured from the start of the file (0-indexed)
            type: Either 'keep', in which case the segment was kept, or other, in which case the segment was dropped.
            segmentDuration: Duration in sec of segment.
    """
    # TODO: Instead of anticipating SI segment indices and adding them to the ftab at the start,
    # we should use the neo header in the SI extractor to add the segment indices to an ftab,
    # or to our segments table, which is confusingly NOT a table of SI segments.
    #
    # Get the experiment frame. This should be for a single probe, and a single stream.
    ftab = subject.get_experiment_frame(
        experiment, alias=alias, stream=stream, ftype="bin", probe=probe
    )
    # Split the experiment frame around the exclusions, using precise sample indices.
    if exclusions is None:
        exclusions = wne.utils.get_dummy_artifacts_table()
    segments = _segment_experiment_frame_for_spikeinterface(
        ftab, exclusions
    )  # These are NOT the segments of a SpikeInterface recording!

    # Take the good segments one by one, create an recording object for each, and save these all in a list
    good_segments = segments[segments["type"] == "keep"]
    recordings = list()
    for segment in good_segments.itertuples():
        extractor = se.SpikeGLXRecordingExtractor(
            segment.gate_dir.parent, stream_id=f"{probe}.{stream}"
        )  # segment.gate_dir.parent is the actual gate directory.
        recording = extractor.select_segments(
            [segment.gate_dir_trigger_file_idx]
        ).frame_slice(
            start_frame=segment.withinFileStartFrame,
            end_frame=segment.withinFileEndFrame,
        )
        recordings.append(recording)

    # Combine the good segments
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

    # We return both recording and segments together, rather than making the available separately,
    # to ensure that you never get a segment table unless it is actually proven to produce a valid extractor object.
    return recording, segments


# TODO: Consider making this a method on wne.Project, or at least a function in wne.sorting
# Although, I don't really like the idea that it uses the alias. But the existing data could be moved to eliminate the alias.
# TODO: This seems more like a sorting utility than a SI utility.
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


# TODO: This seems more like a sorting utility than a SI utility.
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


def load_segments_table_from_sorting(
    project: SGLXProject,
    subject: str,
    experiment: str,
    alias: str,
    probe: str,
    sorting: str,
    return_all_segment_types: bool = False,  # If False, only return segments of type "keep"
) -> pd.DataFrame:
    """Load a sorting's segment file.

    Add a couple useful columns: `segmentExpmtPrbAcqFirstTime`, `segmentExpmtPrbAcqLastTime`
    """  # TODO: Useful for what?
    segment_file = get_sorting_file(
        project, subject, experiment, alias, probe, sorting, "segments.htsv"
    )
    if not segment_file.exists():
        raise FileNotFoundError(f"Segment table not found at {segment_file}.")

    segments = ecephys.utils.read_htsv(segment_file)

    segments["nSegmentSamp"] = (
        segments["withinFileEndFrame"] - segments["withinFileStartFrame"]
    )  # TODO: This column should already be present
    segments["segmentDuration"] = (
        segments["nSegmentSamp"].astype(float).div(segments["imSampRate"])
    )  # TODO: This column should already be present
    segments["segmentExpmtPrbAcqFirstTime"] = segments[
        "expmtPrbAcqFirstTime"
    ] + segments["withinFileStartFrame"].astype(float).div(segments["imSampRate"])
    # For LastTime, we work backwards from expmtPrbAcqLastTime (rather than forward
    # from expmtPrbAcqFirstTime) to ensure that segmentExpmtPrbAcqLastTime <= expmtPrbAcqLastTime
    # This is not the case when working forward due to floating point errors
    # segments["segmentExpmtPrbAcqLastTime"] = (
    #     segments["segmentExpmtPrbAcqFirstTime"] + segments["segmentDuration"]
    # ) # Nope
    segments["segmentExpmtPrbAcqLastTime"] = segments["expmtPrbAcqLastTime"] - (
        segments["nFileSamp"] - segments["withinFileEndFrame"]
    ).astype(float).div(segments["imSampRate"])

    assert np.all(
        segments["segmentExpmtPrbAcqFirstTime"] >= segments["expmtPrbAcqFirstTime"]
    )
    assert np.all(
        segments["segmentExpmtPrbAcqLastTime"] <= segments["expmtPrbAcqLastTime"]
    )

    if return_all_segment_types:
        return segments

    return segments[segments["type"] == "keep"]


# This requires a segment table to have been created and saved to disk.
# It is therefore not general, and is only intended to be used for sorting results.
# TODO: There should be a get_sample2time() function that just takes a segment table directly.
def get_sample2time_from_sorting(
    project: SGLXProject,
    subject: str,
    experiment: str,
    alias: str,
    probe: str,
    sorting: str,
    allow_no_sync_file: bool = False,
    progress_bar: bool = False,  # Seems to have no impact on performance
) -> Callable:
    """Converts sample indices from a SpikeInterface recording to seconds."""
    # Load probe sync table.
    probe_sync_file = project.get_experiment_subject_file(
        experiment, subject, "prb_sync.ap.htsv"
    )
    if not probe_sync_file.exists():
        if allow_no_sync_file:
            logger.info(
                f"Could not find sync table at {probe_sync_file}.\n"
                f"`allow_no_sync_file` == True : Ignoring probe sync in sample2time"
            )
            sync_table = None
        else:
            raise FileNotFoundError(f"No sync file at {probe_sync_file}")
    else:
        sync_table = ecephys.utils.read_htsv(
            probe_sync_file
        )  # Used to map this probe's times to imec0.

    # Load segment table
    segments = load_segments_table_from_sorting(
        project,
        subject,
        experiment,
        alias,
        probe,
        sorting,
        return_all_segment_types=False,
    )  # Used to map SI sorting samples to this probe's times.

    # Get all the good segments (aka the ones in the sorting), in chronological order.
    # Compute which samples in the recording belong to each segment.
    sorted_segments = segments[segments["type"] == "keep"].copy()
    sorted_segments["nSegmentSamples"] = (
        sorted_segments["withinFileEndFrame"] - sorted_segments["withinFileStartFrame"]
    )  # N of sorted samples in each segment

    cum_sorted_samples_by_end = sorted_segments[
        "nSegmentSamples"
    ].cumsum()  # N of sorted samples by the end of each segment
    cum_sorted_samples_by_start = cum_sorted_samples_by_end.shift(
        1, fill_value=0
    )  # N of sorted samples by the start of each segment
    sorted_segments["start_sample"] = (
        cum_sorted_samples_by_start  # First sample index of concatenated recording belonging to each semgent
    )
    sorted_segments["end_sample"] = cum_sorted_samples_by_end
    # TODO: Rename start_sample -> si_start_sample, and end_sample -> si_end_sample?

    # Given a sample number in the SI recording, we can now figure out:
    #   (1) the segment it came from
    #   (2) the file that segment belongs to
    #   (3) how to map that file's times into our canonical timebase.
    # We make a function that does this for an arbitrary array of sample numbers in the SI object, so we can use it later as needed.
    if sync_table is not None:
        sync_table = sync_table.set_index("source")

    def sample2time(s: np.ndarray, progress_bar: bool = progress_bar) -> np.ndarray:
        s = s.astype("float")
        t = np.empty(s.size, dtype="float")
        t[:] = np.nan  # Check a posteriori if we covered all input samples
        iterable = list(sorted_segments.itertuples())
        if progress_bar:
            iterable = tqdm.tqdm(iterable)
        for seg in iterable:
            mask = (s >= seg.start_sample) & (
                s < seg.end_sample
            )  # Mask samples belonging to this segment
            t[mask] = (
                (s[mask] - seg.start_sample) / seg.imSampRate
                + seg.expmtPrbAcqFirstTime
                + seg.withinFileStartFrame / seg.imSampRate
            )  # Convert to number of seconds in this probe's (expmtPrbAcq) timebase
            if sync_table is not None:
                sync_entry = sync_table.loc[
                    seg.fname
                ]  # Get info needed to sync to imec0's (expmtPrbAcq) timebase
                t[mask] = (
                    sync_entry.slope * t[mask] + sync_entry.intercept
                )  # Sync to imec0 (expmtPrbAcq) timebase
        assert not any(np.isnan(t)), (
            "Some of the provided sample indices were not covered by segments \n"
            "and therefore couldn't be converted to time"
        )

        return t

    return sample2time


# TODO: This function needs to be differentiated from load_sglx_inclusions_and_artifacts.
# When would you use this one, and when would you use the other?
# It seems like this one should be used when a sorting already exists, whereas the other
# is used to create a sorting.
# TODO: This function does NOT actually return inclusions and artifacts separately.
# Why both with the second dummy return? Really all we are doing is loading the segments
# table, concerting to times, and then filtering to keep only the segments of type "keep".
def load_sorting_inclusions_and_artifacts(
    t2t: Callable,
    project: SGLXProject,
    sglx_subject: SGLXSubject,
    experiment: str,
    probe: str,
    alias: str,
    sorting: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Query inclusions from segments used in actual sorting
    # segments are in probe timebase and need to be converted to common timebase
    segments = load_segments_table_from_sorting(
        project,
        sglx_subject.name,
        experiment,
        alias,
        probe,
        sorting,
        return_all_segment_types=True,
    ).copy()
    segments = pd.DataFrame(
        {
            "start_time": t2t(segments["segmentExpmtPrbAcqFirstTime"]),
            "end_time": t2t(segments["segmentExpmtPrbAcqLastTime"]),
            "type": segments["type"],
        }
    )  # Raw segment table is in probe timebase

    inclusions = segments.loc[segments["type"] == "keep"]
    artifacts = inclusions.iloc[:0].copy()  # Dummy

    return inclusions, artifacts


def _get_gaps(
    df: pd.DataFrame,
    t1_colname: str = "start_time",
    t2_colname: str = "end_time",
    min_gap_duration_sec: float = 0,
):
    gaps = pd.DataFrame(
        {
            t1_colname: df.iloc[:-1][t2_colname].values,
            t2_colname: df.iloc[1:][t1_colname].values,
        }
    )
    gaps["duration"] = gaps[t2_colname] - gaps[t1_colname]
    return gaps[gaps["duration"] > min_gap_duration_sec]


# TODO: This seems more like a sorting utility, or a wisc_ecephys_tools hypnogram utility, than a SI utility.
# SUS: This function's name does not appear to describe what it does: return all the NoData and/or artifactual periods from a probe.
def load_bouts_to_reconcile_as_hypnogram(
    project: SGLXProject,
    experiment: str,
    sglx_subject: SGLXSubject,
    probe: str,
    source: str,  # TODO: Currently, you have to check first if the source exists. This function should probably do that for you.
    alias: str = "full",
    sorting: str = "sorting",
    min_bout_duration_sec: float = MIN_BOUT_DURATION_SEC,
) -> hyp.FloatHypnogram:
    if source in ["sorting", "ap"]:
        stream = "ap"
    elif source == "lf":
        stream = "lf"
    else:
        raise ValueError(f"Invalid source: {source}")

    # Convert from probe timebase to common timebase asap
    t2t = utils.get_time_synchronizer(
        project,
        sglx_subject,
        experiment,
        probe=probe,
        stream=stream,
    )

    if source == "sorting":
        inclusions, artifacts = load_sorting_inclusions_and_artifacts(
            t2t,
            project,
            sglx_subject,
            experiment,
            probe,
            alias,
            sorting,
        )

    elif source in ["lf", "ap"]:
        inclusions, artifacts = utils.load_sglx_inclusions_and_artifacts(
            t2t,
            project,
            sglx_subject,
            experiment,
            probe,
            alias,
            stream,
        )

    # Infer "NoData" hypnogram from inclusions
    # 1-sample imprecision
    no_data = _get_gaps(
        inclusions,
        t1_colname="start_time",
        t2_colname="end_time",
    )
    no_data["state"] = "NoData"
    no_data_hg = hyp.FloatHypnogram(no_data)

    # Get "artifacts" hypnogram: "type" column now becomes "state"
    artifacts = artifacts.rename(columns={"type": "state"})
    artifacts["duration"] = artifacts["end_time"] - artifacts["start_time"]
    artifacts_hg = hyp.FloatHypnogram(artifacts)

    # Reconcile NoData & artifacts
    return hyp.FloatHypnogram(
        no_data_hg.reconcile(artifacts_hg, how="other")
        .keep_longer(min_bout_duration_sec)
        .reset_index(drop=True)
    )


# TODO: This should be in wisc_ecephys_tools, since it uses the ephyviewer edits.
# TODO: There does need to be a function somewhere in wne that reconciles a consoldiated
# visbrain hypnogram with consolidated artifacts.
# TODO: The sources argument dictates the sources of the NoData and artifacts.
# The current options are "sorting", "ap", and "lf", but are misleading and ambiguous.
# For example, "lf" will pull NoData from the SGLX filetable for each specified probe,
# and artifacts from the project's consoldiated artifact files for each specified probe.
# So these are really separate sources.
# TODO: "sorting" is unable to distinguish between NoData and artifacts, because
# it is not possible to tell from the sorting segments table whether a segment was
# NoData or artifactual, I think? And so artifactual periods will be labled as nodata.
# One result of this is that the order of `sources` actually matters, because the last
# source can determine if a period is marked as NoData or artifactual.
# TODO: What if "sorting" is specified, but not all probes have a sorting?
# Or, what if an artifact file is not found for a probe-stream combination?
# TODO:
# Maybe:
# - `include_lf_sglx_filetable_nodata=True`
# - `include_lf_consolidated_artifacts=True`
# - `include_ap_sglx_filetable_nodata=True`
# - `include_ap_consolidated_artifacts=True`
# - `include_sorting_nodata=True`
# - `include_ephyviewer_edits=True`
def load_reconciled_float_hypnogram(
    project: SGLXProject,
    experiment: str,
    sglx_subject: SGLXSubject,
    probes: list[str],
    sources: list[str],  #
    reconcile_ephyviewer_edits: bool = True,  # TODO: Why is this optional, when it is literally in the function's name?
    simplify: bool = True,
    alias="full",
    sorting="sorting",
) -> hyp.FloatHypnogram:
    """Load FloatHypnogram reconciled with LF/AP/sorting artifacts & NoData.

    Favor using this function, rather than load_raw_float_hypnogram, for
    actual analyses! It ensures that :
        - the probes' actual NoData bouts and # TODO: Actual as opposed to...? Where does the alleged discrepancy come from?
        - the probes' artifacts
        - Manual edits made post-hoc in ephyviewer
    are incorporated

    This is not guaranteed to be the case with the load_raw_float_hypnogram,
    in particular since:
        - SGLX files and artifacts may vary across streams/probes
        - Some bouts may have been excluded from a sorting # TODO: Why?

    If probes and sources are not passed (e.g. empty lists), this function effectively
    is just for loading the ephyviewer edits and/or cleaning.

    Parameters:
    ===========
    project: Project
        Used to load sorting segments, sync table, and LF artifacts
    experiment: str
    subject: SGLXSubject
    probes: list[str]
        Probes for which we load bouts to reconcile with raw hypnogram
    sources: list[str]
        Sources must be one of ["ap", "lf", "sorting"].
        For "lf" and "ap" source, the NoData bouts are inferred from the sglx filetable,
        and the artifacts are loaded from the project's default consolidated
        artifact file.  For "sorting" source, NoData bouts are loaded from the
        sorting segments table.
        # TODO: How are these related? Is the sorting segments table always a superset of the consolidated artifact file?
    simplify: bool
        Passed to load_raw_float_hypnogram. Simplifies states from raw float hypnogram
    alias: str
        Alias used for sorting. Used only when querying "sorting" source.
    sorting: str
        Name of sorting. Used only when querying "sorting" source
    """
    SOURCES = ["sorting", "ap", "lf"]
    if not set(sources) <= set(SOURCES):
        raise ValueError(
            f"Invalid value in `sources` argument. The following sources are recognized: `{SOURCES}`"
        )  # TODO: This is not true. As written, sources=[] will also pass this test. Is that intended, or a bug?
    hg = wne.utils.load_raw_float_hypnogram(
        project,
        experiment,
        sglx_subject.name,
        simplify=simplify,
    )
    if reconcile_ephyviewer_edits:
        hg = hg.reconcile(
            wne.utils.load_ephyviewer_hypnogram_edits(
                project, experiment, sglx_subject.name, simplify=simplify
            ),
            how="other",
        )

    # Only keep bouts that are artifact-free and have data on EVERY probe. The "most conservative" hypnogram, if you will.
    # Any period covered by this hypnogram is guaranteed to be artifact-free and have data on every probe.
    # SUS: If either sources or probes is an empty list, this for-loop will be bypassed entirely. Is that intended?
    # Almost certainly not. Both sources and probes should be required non-empty.
    for source, probe in itertools.product(sources, probes):
        hg = hg.reconcile(
            load_bouts_to_reconcile_as_hypnogram(
                project,
                experiment,
                sglx_subject,
                probe,
                source,
                alias=alias,
                sorting=sorting,
            ),
            how="other",
        )

    return hyp.FloatHypnogram.clean(hg.reset_index(drop=True))


def load_singleprobe_sorting(
    sglxSortingProject: SGLXProject,
    sglxSubject: SGLXSubject,
    experiment: str,
    probe: str,
    alias: str = "full",
    sorting: str = "sorting",
    postprocessing: str = "postpro",
    wneAnatomyProject: Optional[SGLXProject] = None,
    allow_no_sync_file=False,
) -> units.SpikeInterfaceKilosortSorting:
    if sorting is None:
        sorting = "sorting"
    if postprocessing is None:
        postprocessing = "postpro"

    # Get function for converting SI samples to imec0 timebase
    sample2time = get_sample2time_from_sorting(
        sglxSortingProject,
        sglxSubject.name,
        experiment,
        alias,
        probe,
        sorting,
        allow_no_sync_file=allow_no_sync_file,
    )

    # Load extractor
    extractor = sglxSortingProject.get_kilosort_extractor(
        sglxSubject.name,
        experiment,
        probe,
        alias=alias,
        sorting=sorting,
        postprocessing=postprocessing,
    )

    # TODO: Why was this removed, and should it be restored?
    # extractor = units.si_ks_sorting.fix_isi_violations_ratio(extractor)

    # Add anatomy to the extractor, if available.
    if wneAnatomyProject is None:
        wneAnatomyProject = sglxSortingProject

    anatomy_file = wneAnatomyProject.get_experiment_subject_file(
        experiment, sglxSubject.name, f"{probe}.structures.htsv"
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
    sglxSortingProject: SGLXProject,
    sglxSubject: SGLXSubject,
    experiment: str,
    probes: list[str],
    alias: str = "full",
    sortings: dict[str, str] = None,
    postprocessings: dict[str, str] = None,
    wneAnatomyProject: Optional[SGLXProject] = None,
    allow_no_sync_file=False,
) -> units.MultiSIKS:
    if sortings is None:
        sortings = {prb: None for prb in probes}
    if postprocessings is None:
        postprocessings = {prb: None for prb in probes}

    return units.MultiSIKS(
        {
            probe: load_singleprobe_sorting(
                sglxSortingProject,
                sglxSubject,
                experiment,
                probe=probe,
                alias=alias,
                sorting=sortings[probe],
                postprocessing=postprocessings[probe],
                wneAnatomyProject=wneAnatomyProject,
                allow_no_sync_file=allow_no_sync_file,
            )
            for probe in probes
        }
    )
