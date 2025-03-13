import logging
from typing import Callable, Optional

import numpy as np
import pandas as pd
import spikeinterface as si
import spikeinterface.extractors as se
import tqdm

import ecephys.utils
from ecephys import wne
from ecephys.wne.sglx.project import SGLXProject
from ecephys.wne.sglx.subject import SGLXSubject

logger = logging.getLogger(__name__)


def _segment_experiment_frame_for_spikeinterface(
    subject_ftab: pd.DataFrame, exclusions: pd.DataFrame
) -> pd.DataFrame:
    """Split an experiment frame for a single subject, probe, stream, and filetype around a set of periods to exclude.
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
    # Get the experiment frame. This should be for a single probe, and a single stream.
    ftab = subject.get_experiment_frame(
        experiment, alias=alias, stream=stream, ftype="bin", probe=probe
    )
    # Split the experiment frame around the exclusions, using precise sample indices.
    if exclusions is None:
        exclusions = wne.utils.get_dummy_artifacts_table()
    segments = _segment_experiment_frame_for_spikeinterface(ftab, exclusions)

    # Take the good segments one by one, create an recording object for each, and save these all in a list
    good_segments = segments[segments["type"] == "keep"]
    recordings = list()
    for segment in good_segments.itertuples():
        extractor = se.SpikeGLXRecordingExtractor(
            segment.gate_dir, stream_id=f"{probe}.{stream}"
        )
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


def load_segments_table(
    project: SGLXProject,
    subject: str,
    experiment: str,
    alias: str,
    probe: str,
    sorting: str,
    return_all_segment_types: bool = False,
) -> pd.DataFrame:
    """Load a sorting's segment file.

    Add a couple useful columns: `nSegmentSamp`, `segmentDuration`, `segmentExpmtPrbAcqFirstTime`, `segmentExpmtPrbAcqLastTime`
    """
    segment_file = (
        project.get_alias_subject_directory(experiment, alias, subject)
        / f"{sorting}.{probe}"
        / "segments.htsv"
    )
    if not segment_file.exists():
        raise FileNotFoundError(f"Segment table not found at {segment_file}.")

    segments = ecephys.utils.read_htsv(segment_file)

    segments["nSegmentSamp"] = (
        segments["withinFileEndFrame"] - segments["withinFileStartFrame"]
    )
    segments["segmentDuration"] = (
        segments["nSegmentSamp"].astype(float).div(segments["imSampRate"])
    )
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
def get_sample2time(
    project: SGLXProject,
    subject: str,
    experiment: str,
    alias: str,
    probe: str,
    sorting: str,
    allow_no_sync_file: bool = False,
    progress_bar: bool = False,
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
    segments = load_segments_table(
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
