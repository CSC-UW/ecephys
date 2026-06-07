# TODO: This file should probably be broken into separate modules.
import logging
import pathlib
from typing import Callable, Optional

import numba
import numpy as np
import pandas as pd

import ecephys.utils
from ecephys.sglx import file_mgmt
from ecephys.wne import constants

from . import sessions
from .project import SGLXProject
from .subject import SGLXSubject

logger = logging.getLogger(__name__)


class UnfinalizedRecordingError(ValueError):
    """Raised when an absolute (acquisition-clock) time is required but unknown.

    Occurs for an un-finalized SpikeGLX recording whose .meta lost `firstSample`
    (see ``ecephys.sglx.repair_metadata``): its ``expmtPrbAcqFirstTime`` is NaN, so
    within-file times cannot be placed on the experiment-acquisition (or canonical)
    timebase. Within-file analysis is still possible; absolute/synced times are not.
    """


def require_acq_time(file_row) -> float:
    """Return ``expmtPrbAcqFirstTime`` for a file row, or raise if it is unknown.

    Use in place of ``file_row.expmtPrbAcqFirstTime`` wherever an absolute time is
    needed, so an un-finalized recording fails loudly with an actionable message
    instead of silently producing NaN-stamped outputs. Pure pass-through (returns
    the value unchanged) when the offset is known.
    """
    t0 = file_row.expmtPrbAcqFirstTime
    if pd.isna(t0):
        name = getattr(getattr(file_row, "path", None), "name", file_row)
        raise UnfinalizedRecordingError(
            f"{name}: acquisition offset (expmtPrbAcqFirstTime) is unknown -- this "
            "recording's .meta lost `firstSample` (un-finalized/crashed). Cannot place "
            "times on the experiment-acquisition or canonical timebase. Re-finalize "
            "the meta (ecephys.sglx.repair_metadata) or exclude this session."
        )
    return t0


def get_sglx_file_counterparts(
    project: SGLXProject,
    subject: str,
    paths: list[pathlib.Path],
    extension: str,
    remove_probe: bool = False,
    remove_stream: bool = False,
) -> list[pathlib.Path]:
    """Get counterparts to SpikeGLX raw data files.

    Counterparts are mirrored at the project's subject directory, and likely
    have different suffixes than the original raw data files.

    Parameters:
    -----------
    project_name: str
        From projects.yaml
    subject_name: str
        Subject's name within this project, i.e. subject's directory name.
    paths: list of pathlib.Path
        The raw data files to get the counterparts of.
    extension:
        The extension to replace .bin or .meta with. See `replace_ftype`.

    Returns:
    --------
    list of pathlib.Path
    """
    counterparts = sessions.mirror_raw_data_paths(
        project.get_subject_directory(subject), paths
    )  # Mirror paths at the project's subject directory
    counterparts = [
        file_mgmt.replace_ftype(p, extension, remove_probe, remove_stream)
        for p in counterparts
    ]
    return ecephys.utils.remove_duplicates(counterparts)


def get_time2time(
    experiment_sync_table: pd.DataFrame,
    experiment_probe_ftable: pd.DataFrame,
    binfile: Optional[pathlib.Path] = None,
    extrapolate: bool = False,
) -> Callable[[np.ndarray], np.ndarray]:
    assert len(experiment_probe_ftable["probe"].unique()) == 1, (
        "Cannot generate a time2time function without knowing the probe"
    )
    experiment_sync_table = experiment_sync_table.set_index("source")

    if binfile is not None:
        # If we know the file a-priori, we can give a maximally precise time2time function
        def file_time2time(t1):
            sync_entry = experiment_sync_table.loc[binfile.name]
            return sync_entry.slope * t1 + sync_entry.intercept

        return file_time2time

    else:
        # If we don't know the binfile a-priori, our time-to-time function has to infer it.
        # WARNING: Because of file overlap, this method of assigning times to files is imperfect! Use per-file sync for maximum precision!
        def experiment_time2time(t1):
            # An un-finalized recording has NaN expmt windows, so the range mask
            # below can't place times in it. If it is the ONLY file for this probe
            # there is nothing to disambiguate -- every time maps to it, and the
            # unknown offset is irrelevant (slope/intercept absorb it).
            if experiment_probe_ftable["expmtPrbAcqFirstTime"].isna().any():
                if len(experiment_probe_ftable) == 1:
                    sync_entry = experiment_sync_table.loc[
                        experiment_probe_ftable.iloc[0].path.name
                    ]
                    return sync_entry.slope * t1 + sync_entry.intercept
                raise UnfinalizedRecordingError(
                    "Cannot infer per-file membership: this probe mixes files with "
                    "and without a known acquisition window (NaN expmtPrbAcqFirstTime). "
                    "Pass an explicit `binfile=` for per-file sync, or exclude the "
                    "un-finalized session."
                )
            t2 = np.full_like(t1, fill_value=np.nan)
            for file in experiment_probe_ftable.itertuples():
                mask = (t1 >= file.expmtPrbAcqFirstTime) & (
                    t1 <= file.expmtPrbAcqLastTime + (1 / file.imSampRate)
                )  # Mask samples belonging to this file
                sync_entry = experiment_sync_table.loc[
                    file.path.name
                ]  # Get info needed to sync to the reference probe's (expmtPrbAcq) timebase
                t2[mask] = (
                    sync_entry.slope * t1[mask] + sync_entry.intercept
                )  # Sync to the reference probe's (expmtPrbAcq) timebase
            is_nan = np.isnan(t2)
            if any(is_nan):
                msg = "Some of the provided times were not covered by the original recording and therefore can't be converted unambiguously."
                if extrapolate:
                    logger.warning(msg + " Using sync info from the nearest file.")
                    allowed_times = experiment_probe_ftable[
                        ["expmtPrbAcqFirstTime", "expmtPrbAcqLastTime"]
                    ].values.flatten()
                    allowed_files = experiment_probe_ftable[
                        ["path", "path"]
                    ].values.flatten()
                    for ix_t in np.where(is_nan)[0]:
                        nearest_allowed = ecephys.utils.find_nearest(
                            allowed_times, t1[ix_t]
                        )
                        nearest_fname = allowed_files[nearest_allowed].name
                        sync_entry = experiment_sync_table.loc[nearest_fname]
                        t2[ix_t] = sync_entry.slope * t1[ix_t] + sync_entry.intercept
                else:
                    raise ValueError(msg)

            return t2

        return experiment_time2time


def get_time_synchronizer(
    sync_project: SGLXProject,
    sglx_subject: SGLXSubject,
    experiment: str,
    stream: Optional[str] = None,
    probe: Optional[str] = None,
    binfile: Optional[pathlib.Path] = None,
    extrapolate: bool = False,
) -> Callable[[np.ndarray], np.ndarray]:
    if binfile is not None:
        (_, _, _, probe_, stream_, _) = file_mgmt.parse_sglx_fname(binfile.name)
        probe = probe_ if probe is None else probe
        assert probe == probe_, "Mismatch between provided probe and binfile"
        stream = stream_ if stream is None else stream
        assert stream == stream_, "Mismatch between provided stream and binfile"
    assert probe is not None, "Must provide probe"
    assert stream is not None, "Must provide stream"
    experiment_probe_ftable = sglx_subject.get_experiment_frame(
        experiment, ftype="bin", stream=stream, probe=probe
    )
    experiment_sync_table = ecephys.utils.read_htsv(
        sync_project.get_experiment_subject_file(
            experiment, sglx_subject.name, constants.SYNC_FNAME_MAP[stream]
        )
    )
    return get_time2time(
        experiment_sync_table, experiment_probe_ftable, binfile, extrapolate
    )


def load_consolidated_artifacts(
    project: SGLXProject,
    experiment: str,
    subject: str,
    probe: str,
    stream: str,
    simplify: bool = True,
):
    """
    All times are already in the canonical timebase. No sync/conversion is needed.
    If a consolidated artifact file does not exist, an empty table is returned.
    """
    artifacts_path = project.get_experiment_subject_file(
        experiment,
        subject,
        f"{probe}.{stream}.{constants.Files.ARTIFACTS}",
    )
    if artifacts_path.exists():
        artifacts = ecephys.utils.read_htsv(artifacts_path).loc[
            :, ["start_time", "end_time", "type"]
        ]
    else:
        artifacts = pd.DataFrame([], columns=["start_time", "end_time", "type"])

    if simplify:
        return artifacts.replace(constants.SIMPLIFIED_ARTIFACTS)

    return artifacts


def create_slice_table_for_spikeinterface(
    subject_ftab: pd.DataFrame,
    exclusions: pd.DataFrame,
    return_dropped_slices: bool = False,
) -> pd.DataFrame:
    """Split an experiment frame for a single subject, probe, stream, and filetype
    around a set of periods to excise/exclude. Though used to create a spikeinterface
    ConcatenateSegmentRecording from FrameSliceRecordings, it is in principle
    independent of spikeinterface.

    Parameters
    ==========
    subject_ftab: pd.DataFrame
        The experiment frame for a single subject, probe, stream, and filetype.
        There will be 1 row per file.
        The columns used are `path`, `nFileSamp`, and `imSampRate`.
    exclusions: pd.DataFrame
        Specify which parts of the recording to drop. We slice such that the first and
        last samples of each exclusion are NOT included in the returned recording.
        Required columns are `fname`, `withinFileStartTime`, `withinFileEndTime`.
        A `type` column is expected, but not required.
    return_dropped_slices: bool
        If True, return all slices, including those to be dropped.

    Returns
    =======
    slice_table: pd.DataFrame
        A slice table where each row is a slice of data to keep or drop, sorted in
        chronological order. There may be multiple rows per file, if exclusions
        were specified that split the file into multiple slices.
        A `type` columns indicates whether the slice is to be kept or dropped. If kept,
        the value will be "keep". If dropped, the value will be whatever was specified
        in the `exclusions` table, or null if not specified.


    Notes
    =====
    Slices do NOT correspond to SpikeInterface segments.
    The equivalence between SpikeInterface objects and our objects is, roughly:
        - SI SpikeGLXRecordingExtractor <-> SGLX gate directory, with probe subdirectory
          if folder-per-probe orgnization is used.
        - SI Segment <-> A single binfile.
        - SI FrameSliceRecording <-> An entry in the slice table.
    """
    # Check that the exclusions table has the required columns.
    # TODO: The exclusion schema should be defined elsewhere, or more formally.
    #       Note that it will differ slightly from the slice_table schema.
    required_exclusion_cols = [
        "fname",
        "withinFileStartTime",
        "withinFileEndTime",
        "type",
    ]
    assert all([c in exclusions.columns for c in required_exclusion_cols]), (
        f"Exclusions require all of the following columns: `{required_exclusion_cols}`"
    )

    slices = list()

    # For each file in the experiment, split it if necessary.
    # If not, just create a slice that is the entire file.
    for file in subject_ftab.itertuples():
        # Get the exclusions pertaining to this file.
        in_file = exclusions["fname"] == file.path.name

        # For the exclusions pertaining to this file, convert their definition in
        # seconds to precise sample indices, and clip these estimates so that sample
        # indices don't extend beyond the ends of the file.
        exclusions.loc[in_file, "withinFileStartFrame"] = (
            (exclusions.loc[in_file, "withinFileStartTime"] * file.imSampRate)
            .astype(int)
            .clip(0, file.nFileSamp)
        )
        exclusions.loc[in_file, "withinFileEndFrame"] = (
            (exclusions.loc[in_file, "withinFileEndTime"] * file.imSampRate)
            .astype(int)
            .clip(0, file.nFileSamp)
        )

        # Do the actual splitting of the entire file around the exclusions
        whole_file_slice = pd.DataFrame(
            {
                "withinFileStartFrame": [0],
                "withinFileEndFrame": [file.nFileSamp],
                "type": "keep",
            }
        )  # Keep the whole file by default, if no exclusions exist.
        file_exclusions = exclusions.loc[
            in_file, ["withinFileStartFrame", "withinFileEndFrame", "type"]
        ]
        file_slices = ecephys.utils.pandas.reconcile_labeled_intervals(
            file_exclusions,
            whole_file_slice,
            "withinFileStartFrame",
            "withinFileEndFrame",
        ).drop(columns="delta")
        file_slices["fname"] = file.path.name

        # `reconcile_labeled_intervals()` considers intervals to be open-ended, so that
        # (a, b) and (b, c) are considered NON-overlapping (usual python slicing).
        # This means that up to this point the end sample of an exclusion will be part
        # of the next (kept) slice. In order to be conservative, we correct each bad
        # slice followed by a good slice to include its last sample. For example,
        # if (a, b) and (b, c) are bad slice, and (c, d) is a good slice, the new
        # slices will be (a, b+1), (b+1, c), (c+1, d).
        keep = file_slices["type"] == "keep"
        frames_to_shift = np.intersect1d(
            file_slices[~keep]["withinFileEndFrame"].values,
            file_slices["withinFileStartFrame"].values,
        )  # End of each bad slice followed by another slice (excludes the last one)
        i = file_slices["withinFileEndFrame"].isin(frames_to_shift)
        j = file_slices["withinFileStartFrame"].isin(frames_to_shift)
        file_slices.loc[i, "withinFileEndFrame"] += 1
        file_slices.loc[j, "withinFileStartFrame"] += 1

        # Sanity checks, ensuring that every sample in the file is accounted for.
        assert file_slices["withinFileStartFrame"].min() == 0, (
            "Something went wrong when splitting file around exclusions."
        )
        assert file_slices["withinFileEndFrame"].max() == (file.nFileSamp), (
            "Something went wrong when splitting file around exclusions."
        )
        assert (
            file_slices["withinFileEndFrame"] - file_slices["withinFileStartFrame"]
        ).sum() == file.nFileSamp, (
            "Something went wrong when splitting file around exclusions."
        )

        # Checks passed, add these slices to the overall list.
        slices.append(file_slices)

    slice_table = pd.concat(slices, ignore_index=True).astype(
        {"withinFileStartFrame": int, "withinFileEndFrame": int}
    )
    slice_table = slice_table.rename(columns={"type": "sliceType"})

    # Add file metadata to the slice table.
    subject_ftab = subject_ftab.copy()  # Do not modify input dataframe in-place.
    subject_ftab["fname"] = subject_ftab["path"].apply(lambda x: x.name)
    slice_table = slice_table.merge(subject_ftab, on="fname")

    if return_dropped_slices:
        return slice_table

    return slice_table[slice_table["sliceType"] == "keep"].reset_index(drop=True)


def add_sample2time_columns(
    slice_table: pd.DataFrame, sync_table: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Add columns to slice_table to support vectorized sample2time conversion.

    Parameters
    ==========
    slice_table:
        A dataframe with columns:
        - 'fname'
        - 'withinFileStartFrame'
        - 'withinFileEndFrame'
        - 'sliceType'
        - 'imSampRate'
        - 'expmtPrbAcqFirstTime'
        ...describing the slices that were concatenated to form the recording.
        NO EXCISED SLICES SHOULD BE PRESENT!
        Each row is a slice. There may be mutiple slices from the same file,
        if excisions/exclusions were made. In the case of no excisions/exclusions:
        1. The `slice_table` is the `experiment_probe_ftable`.
        2. `withinFileStartFrame` is always 0 for every slice.
        3. `withinFileEndFrame` is always `nSliceSamples` / `nFileSamp`.
        See `create_slice_table_for_spikeinterface()`.
    sync_table:
        A dataframe with columns 'source', 'slope', and 'intercept', mapping times
        from each source file to the canonical timebase. 'source' is a filename.

    Returns
    =======
    pd.DataFrame
        Copy of input slice_table with additional columns:
        - n_slice_samples: Number of samples in this slice.
        - start_sample: Start sample index of this slice in the concatenated recording.
        - end_sample: End sample index of this slice in the concatenated recording.
        - time_offset: Unsynchronized start time of this slice.
        - sync_slope: Slope for synchronizing this slice's time to the canonical timebase.
        - sync_intercept: Intercept for synchronizing this slice's time to the canonical timebase.

    Notes
    =====
    The basic idea here is that, given a sample number in a spliced/concatenated
    recording, we can figure out:
    1. The slice it came from.
    2. The file that slice comes from.
    3. How to map that file's times into our canonical timebase.

    """
    slices = slice_table.copy()  # Do not modify input dataframe in-place.
    assert all(slices["sliceType"] == "keep"), (
        "`slice_table` must not contain excised/excluded slices."
    )

    # Absolute (acquisition-clock) offsets are required to build sample->time
    # columns. An un-finalized recording (lost `firstSample`) has NaN offsets, so
    # fail loudly rather than emit NaN-stamped times.
    if slices["expmtPrbAcqFirstTime"].isna().any():
        bad = sorted(slices.loc[slices["expmtPrbAcqFirstTime"].isna(), "fname"].unique())
        raise UnfinalizedRecordingError(
            f"Cannot build sample->time columns: {bad} have unknown acquisition "
            "offsets (un-finalized recording). Re-finalize their metas "
            "(ecephys.sglx.repair_metadata) or exclude the session."
        )

    # Get the number of samples in each slice
    slices["n_slice_samples"] = (
        slices["withinFileEndFrame"] - slices["withinFileStartFrame"]
    )

    # Get the start and end sample indices of each slice in the concatenated recording
    slices["end_sample"] = slices["n_slice_samples"].cumsum()
    slices["start_sample"] = slices["end_sample"].shift(1, fill_value=0)

    # Get the unsynchronized start time of each slice
    slices["time_offset"] = (
        slices["expmtPrbAcqFirstTime"]
        + slices["withinFileStartFrame"] / slices["imSampRate"]
    )

    if sync_table is not None:
        sync_table = sync_table.set_index("source")
        slices["sync_slope"] = slices["fname"].map(sync_table["slope"])
        slices["sync_intercept"] = slices["fname"].map(sync_table["intercept"])
    else:
        slices["sync_slope"] = 1.0
        slices["sync_intercept"] = 0.0

    return slices


def get_sample2time(slice_table: pd.DataFrame) -> Callable:
    """For a concatenated recording (e.g. a spikeinterface ConcatenateSegmentRecording)
    built up from slices of other recordings, with possible excisions/exclusions
    occuring before concatenation, get a function that converts sample indices from this
    concatenated recording into synchronized times from the canonical timebase.

    Note that, this function makes sense *if you don't know the samples in advance*.
    If you are precomputing all times for e.g. a set of slices, or a whole recording,
    there are MUCH faster ways to do this, because you don't need to infer which slice
    a sample came from! See slice_table2times().

    Parameters
    ==========
    slice_table:
        A dataframe with columns:
        - start_sample
        - end_sample
        - imSampRate
        - time_offset
        - sync_slope
        - sync_intercept
        ...describing the slices that were concatenated to form the recording.
        NO EXCISED SLICES SHOULD BE PRESENT!
        Each row is a slice. There may be mutiple slices from the same file,
        if excisions/exclusions were made. In the case of no excisions/exclusions:
        1. The `slice_table` is the `experiment_probe_ftable`.
        2. `withinFileStartFrame` is always 0 for every slice.
        3. `withinFileEndFrame` is always `nSliceSamples` / `nFileSamp`.
        See `create_slice_table_for_spikeinterface()` and `add_sample2time_columns()`.

    Returns
    =======
    sample2time:
        A function that takes a SORTED array of sample indices from the concatenated
        recording, and returns an array of times in seconds from the canonical timebase.

    Notes
    =====

    Implementation details matter a lot here. On the first run ("cold"), NumPy must:

      1. Compile ufuncs - NumPy's universal functions (like >=, &, indexed assignment)
         are lazily compiled/optimized on first use.
      2. Allocate memory - Creating temporary boolean arrays (mask) of any serious size
         repeatedly triggers OS-level memory allocation.
      3. Page faults - The OS must map physical memory pages for newly allocated arrays.

    After the first run ("hot"), the compiled code is cached, and the OS has already
    mapped memory pages (or they're in cache), so subsequent runs are much faster.

    This function could (used to) be implemented in such a way that it could take
    unsorted sample indices, but the cost of this is so high on cold runs, and the
    advantages so few, that it should not be done. The new implementaitons are at least
    3-5x faster.
    """
    # Precompute small arrays.
    # Indexing into these is faster than indexing into df[col].values, probably
    # because of compiler optimizatons.
    _start = slice_table["start_sample"].to_numpy()
    _end = slice_table["end_sample"].to_numpy()
    _dt = (1.0 / slice_table["imSampRate"]).to_numpy()
    _t0 = slice_table["time_offset"].to_numpy()
    _slope = slice_table["sync_slope"].to_numpy()
    _intercept = slice_table["sync_intercept"].to_numpy()

    def sample2time(s: np.ndarray) -> np.ndarray:
        # Runtimes on discontinuous samples:
        # 1e7 samples: ~0.4s (max 1.0s) whether hot or cold.
        # 1e8 samples: ~4.2s (max 11.5s) whether hot or cold.
        # 2e8 samples: ~8.1s whether hot or cold.
        # 8e8 samples: ~40.s whether hot or cold.

        # Find which slice each sample belongs to
        idx = np.searchsorted(_end, s, side="right")  # Slice indices
        return _slope[idx] * ((s - _start[idx]) * _dt[idx] + _t0[idx]) + _intercept[idx]

    def _sample2time_memory_efficient(s: np.ndarray) -> np.ndarray:
        # Runtimes on discontinuous samples:
        # 1e7 samples: ~0.4-0.9s whether hot or cold.
        # 1e8 samples: ~13s hot or cold (max 55.1s cold).
        # 2e8 samples: ~9.0s whether hot or cold.
        # 8e8 samples: ~36.0s whether hot or cold.
        # Thus, this function may be faster and use less memory, though its performance
        # seems more variable, and theoretically the other implementation should win.

        # Find which slice each sample belongs to
        idx = np.searchsorted(_end, s, side="right")  # Slice indices

        # Compute in-place to avoid intermediate arrays
        # t = slope * ((s - start) * dt + t0) + intercept
        t = s - _start[idx]
        t = t.astype(np.float64)  # In-place cast to float64
        t *= _dt[idx]  # In-place multiply
        t += _t0[idx]  # In-place add
        t *= _slope[idx]  # In-place multiply
        t += _intercept[idx]  # In-place add
        return t

    return sample2time


@numba.njit(parallel=True)
def _slice_table2times_kernel(start, ns, dt, t0, slope, intercept):
    """Numba JIT kernel - parallel loop, minimal memory."""
    n_samples = ns.sum()
    t = np.empty(n_samples, dtype=np.float64)

    for i in numba.prange(len(start)):
        for j in range(ns[i]):
            t[start[i] + j] = slope[i] * (t0[i] + j * dt[i]) + intercept[i]

    return t


def slice_table2times(slice_table: pd.DataFrame) -> np.ndarray:
    """For a concatenated recording (e.g. a spikeinterface ConcatenateSegmentRecording)
    built up from slices of other recordings, with possible excisions/exclusions
    occuring before concatenation, get a function that converts the slice table into
    the recording's synchronized times from the canonical timebase.

    Parameters
    ==========
    slice_table:
        A dataframe with columns:
        - start_sample
        - end_sample
        - imSampRate
        - time_offset
        - sync_slope
        - sync_intercept
        ...describing the slices that were concatenated to form the recording.
        NO EXCISED SLICES SHOULD BE PRESENT!
        Each row is a slice. There may be mutiple slices from the same file,
        if excisions/exclusions were made. In the case of no excisions/exclusions:
        1. The `slice_table` is the `experiment_probe_ftable`.
        2. `withinFileStartFrame` is always 0 for every slice.
        3. `withinFileEndFrame` is always `nSliceSamples` / `nFileSamp`.
        See `create_slice_table_for_spikeinterface()` and `add_sample2time_columns()`.

    Returns
    =======
    np.ndarray
        The timestamps for every sample covered by the slice table.

    Notes
    =====
    This function can run in ~2.5s on >3 billion samples (~23 GB) with irregular gaps.

    Testing indicates that it is ~3x faster than a numba-free approach that loops
    over the slice table and computes each slice's time 1-by-1.

    It is ~10x faster than the fully vectorized approach, probably because when arrays
    get so large, the loop approach effectively implements chunking, and numpy may be
    parellelizing the loop iterations under the hood.
    """
    _ns = slice_table["n_slice_samples"].to_numpy()
    _dt = (1.0 / slice_table["imSampRate"]).to_numpy()
    _t0 = slice_table["time_offset"].to_numpy()
    _slope = slice_table["sync_slope"].to_numpy()
    _intercept = slice_table["sync_intercept"].to_numpy()

    # Compute relative start positions within output array
    _end = np.cumsum(_ns)
    _start = np.concatenate([[0], _end[:-1]])

    return _slice_table2times_kernel(_start, _ns, _dt, _t0, _slope, _intercept)
