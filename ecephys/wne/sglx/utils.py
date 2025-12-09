# TODO: This file should probably be broken into separate modules.
import logging
import pathlib
from typing import Callable, Optional

import numpy as np
import pandas as pd

import ecephys.utils
from ecephys.sglx import file_mgmt
from ecephys.wne import constants

from . import sessions
from .project import SGLXProject
from .subject import SGLXSubject

logger = logging.getLogger(__name__)


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


def _get_sample2time(
    slice_table: pd.DataFrame, sync_table: pd.DataFrame | None = None
) -> Callable:
    """For a concatenated recording (e.g. a spikeinterface ConcatenateSegmentRecording)
    built up from slices of other recordings, with possible excisions/exclusions
    occuring before concatenation, get a function that converts sample indices from this
    concatenated recording into synchronized times from the canonical timebase.

    Parameters
    ==========
    slice_table:
        A dataframe with columns 'fname', 'withinFileStartFrame', 'withinFileEndFrame',
        'imSampRate', and 'expmtPrbAcqFirstTime', describing the slices that were
        concatenated to form the recording. NO EXCISED SLICES SHOULD BE PRESENT!
        Each row is a slice. There may be mutiple slices from the same file,
        if excisions/exclusions were made. See Notes below for more info.
    sync_table:
        A dataframe with columns 'source', 'slope', and 'intercept', mapping times
        from each source file to the canonical timebase. 'source' is a filename.

    Returns
    =======
    sample2time:
        A function that takes an array of sample indices from the concatenated recording,
        and returns an array of times in seconds from the canonical timebase.

    Notes
    =====
    In the case of no excisions/exclusions:
      1. The `slice_table` is the `experiment_probe_ftable`.
      2. `withinFileStartFrame` is always 0 for every slice.
      3. `withinFileEndFrame` is always `nSliceSamples` / `nFileSamp`.
    """
    # Tom added a "type" column to the slice table, which I don't want to demand from
    # users, but we can check for it and soft-warn if non-`keep` values are present.
    if "type" in slice_table.columns and not all(slice_table["type"] == "keep"):
        print("Warning: `slice_table` contains a `type` column with non-`keep` values.")

    # Get the number of samples in each slice
    slice_table["nSliceSamples"] = (
        slice_table["withinFileEndFrame"] - slice_table["withinFileStartFrame"]
    )
    # Get the number of samples kept by the end of each slice
    cum_samples_by_end = slice_table["nSliceSamples"].cumsum()
    # Get the number of samples kept by the start of each slice
    cum_samples_by_start = cum_samples_by_end.shift(1, fill_value=0)
    # First sample index in concatenated recording belonging to each slice
    slice_table["start_sample"] = cum_samples_by_start
    # Last sample index in concatenated recording belonging to each slice
    slice_table["end_sample"] = cum_samples_by_end

    # Given a sample number in the SI recording, we can now figure out:
    #   (1) the slice it came from
    #   (2) the file that slice comes from
    #   (3) how to map that file's times into our canonical timebase.
    # We make a function that does this for an arbitrary array of sample numbers in the SI object, so we can use it later as needed.
    if sync_table is not None:
        sync_table = sync_table.set_index("source")

    def sample2time(s: np.ndarray) -> np.ndarray:
        s = s.astype("float")
        t = np.empty(s.size, dtype="float")
        t[:] = np.nan  # Check a posteriori if we covered all input samples
        for slc in slice_table.itertuples():
            mask = (s >= slc.start_sample) & (
                s < slc.end_sample
            )  # Mask samples belonging to this slice
            t[mask] = (
                (s[mask] - slc.start_sample) / slc.imSampRate
                + slc.expmtPrbAcqFirstTime
                + slc.withinFileStartFrame / slc.imSampRate
            )  # Convert to number of seconds in this probe's (expmtPrbAcq) timebase
            if sync_table is not None:
                sync_entry = sync_table.loc[
                    slc.fname
                ]  # Get info needed to sync to imec0's (expmtPrbAcq) timebase
                t[mask] = (
                    sync_entry.slope * t[mask] + sync_entry.intercept
                )  # Sync to imec0 (expmtPrbAcq) timebase
        assert not any(np.isnan(t)), (
            "Some of the provided sample indices were not covered by slices \n"
            "and therefore couldn't be converted to time"
        )

        return t

    return sample2time


def get_sample2time(
    sync_project: SGLXProject,
    subject: str,
    experiment: str,
    slice_table: pd.DataFrame,
    allow_no_sync_file: bool = False,
) -> Callable:
    """For a concatenated recording (e.g. a spikeinterface ConcatenateSegmentRecording)
    built up from slices of other recordings, with possible excisions/exclusions
    occuring before concatenation, get a function that converts sample indices from this
    concatenated recording into synchronized times from the canonical timebase.

    Parameters
    ==========
    sync_project:
        SGLXProject instance used to locate the sync file.
    subject:
        Subject name.
    experiment:
        Experiment name.
    slice_table:
        A dataframe with columns 'fname', 'withinFileStartFrame', 'withinFileEndFrame',
        'imSampRate', and 'expmtPrbAcqFirstTime', describing the slices that were
        concatenated to form the recording. NO EXCISED SLICES SHOULD BE PRESENT!
        Each row is a slice. There may be mutiple slices from the same file,
        if excisions/exclusions were made. See Notes below for more info.
    allow_no_sync_file:
        If True, proceed without synchronization if the sync file is not found.
        If False, raise FileNotFoundError when sync file is missing.

    Returns
    =======
    sample2time:
        A function that takes an array of sample indices from the concatenated recording,
        and returns an array of times in seconds from the canonical timebase.

    Notes
    =====
    In the case of no excisions/exclusions:
      1. The `slice_table` is the `experiment_probe_ftable`.
      2. `withinFileStartFrame` is always 0 for every slice.
      3. `withinFileEndFrame` is always `nSliceSamples` / `nFileSamp`.
    """
    sync_file = sync_project.get_experiment_subject_file(
        experiment, subject, constants.Files.AP_SYNC
    )
    if not sync_file.exists():
        print(f"Sync table not found at {sync_file}")
        if allow_no_sync_file:
            print("`allow_no_sync_file` == True : Ignoring probe sync in sample2time")
            sync_table = None
        else:
            raise FileNotFoundError(f"No sync file at {sync_file}")
    else:
        sync_table = ecephys.utils.read_htsv(
            sync_file
        )  # Used to map this probe's times to imec0.
    return _get_sample2time(slice_table, sync_table)


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
            t2 = np.full_like(t1, fill_value=np.nan)
            for file in experiment_probe_ftable.itertuples():
                mask = (t1 >= file.expmtPrbAcqFirstTime) & (
                    t1 <= file.expmtPrbAcqLastTime + (1 / file.imSampRate)
                )  # Mask samples belonging to this file
                sync_entry = experiment_sync_table.loc[
                    file.path.name
                ]  # Get info needed to sync to imec0's (expmtPrbAcq) timebase
                t2[mask] = (
                    sync_entry.slope * t1[mask] + sync_entry.intercept
                )  # Sync to imec0 (expmtPrbAcq) timebase
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
