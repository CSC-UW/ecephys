"""
These functions resolve paths to SpikeGLX data, assuming that the data
are described according to the experiments_and_aliases.yaml format.
The data must also be organized and described 'session-style' -- for
more information, see sglx_sessions.py.

The experiments_and_aliases.yaml format allows you to define experiments
in terms of SpikeGLX recording sessions, and to refer to subsets of this
data using 'aliases'.

Aliases contain a list of {'start_file': <...>, 'end_file': <...>} dictionaries
each specifying a continuous subset of data.
"""
import itertools as it
import logging
import re

import numpy as np
import pandas as pd

from ... import sglx
from .sessions import get_session_files_from_multiple_locations

# TODO: Currently, there can be no clean `get_session_directory(subject_name, experiment_name) function,
# because there is no single session directory -- it can be split across AP and LF locations, and we
# don't know which might hold e.g. video files for that session.

logger = logging.getLogger(__name__)


def parse_trigger_stem(stem):
    """Parse recording identifiers from a SpikeGLX style filename stem.
    Because this stem ends with the trigger identifier, we call it a
    'trigger stem'.

    Although this function may seem like it belongs in ecephys.sglx.file_mgmt,
    it really belongs here. This is because the concept of a trigger stem is
    not really used in SpikeGLX, but is used in experiments_and_aliases.yaml
    as a convenient way of specifying file ranges.

    Parameters
    ---------
    stem: str
        The filename stem to parse, e.g. "my-run-name_g0_t1"

    Returns
    -------
    run: str
        The run name, e.g. "my-run-name".
    gate: str
        The gate identifier, e.g. "g0".
    trigger: str
        The trigger identifier, e.g. "t1".

    Examples
    --------
    >>> parse_trigger_stem('3-1-2021_A_g1_t0')
    ('3-1-2021_A', 'g1', 't0')
    """
    x = re.search(r"_g\d+_t\d+\Z", stem)  # \Z forces match at string end.
    run = stem[: x.span()[0]]  # The run name is everything before the match
    gate = re.search(r"g\d+", x.group()).group()
    trigger = re.search(r"t\d+", x.group()).group()

    return (run, gate, trigger)


def get_gate_dir_trigger_file_index(ftab):
    """Get index of trigger file relative to all files of same stream/prb/gate_folder.

    This is relative to files currently present in the directory, so we can't
    just parse trigger index (doesn't work if some files are moved).

    Useful to instantiate spikeinterface extractor objects, since they require
    subselecting segments of interest (ie trigger files) after instantiation.
    See https://github.com/SpikeInterface/spikeinterface/issues/628#issuecomment-1130232542

    """
    ftab["gate_dir"] = ftab.apply(lambda row: row["path"].parent, axis=1)
    for gate_dir, prb, stream, ftype in it.product(
        ftab.gate_dir.unique(),
        ftab.probe.unique(),
        ftab.stream.unique(),
        ftab.ftype.unique()
    ):
        mask = (
            (ftab["gate_dir"] == gate_dir)
            & (ftab["probe"] == prb)
            & (ftab["stream"] == stream)
            & (ftab["ftype"] == ftype)
        )
        mask_n_triggers = int(mask.sum())
        ftab.loc[mask, "gate_dir_n_trigger_files"] = mask_n_triggers
        ftab.loc[mask, "gate_dir_trigger_file_idx"] = np.arange(0, mask_n_triggers)
    
    ftab['gate_dir_n_trigger_files'] = ftab['gate_dir_n_trigger_files'].astype(int)
    ftab['gate_dir_trigger_file_idx'] = ftab['gate_dir_trigger_file_idx'].astype(int)

    return ftab

def get_start_times_relative_to_experiment(ftab, tol=1, method="rigorous"):
    """Get a series containing the time of each file start, in seconds, relative to the beginning of the experiment.

    ftab: pd.DataFrame
        ALL files from the experiment.
    tol: int
        The number of samples that continuous files are allowed to be overlapped or gapped by.
    method: "rigorous" or "approximate"
        How to determine timestamps. Rigorous is fast.
    """
    ftab = ftab.sort_values("fileCreateTime")
    if method == "approximate":
        ftab["tExperiment"] = (
            ftab["fileCreateTime"] - ftab["fileCreateTime"].min()
        ).dt.total_seconds()
        return ftab
    elif method != "rigorous":
        raise ValueError(f"Method {method} not recognized.")

    # Do the rigourus method
    for probe, stream, ftype in it.product(
        ftab["probe"].unique(), ftab["stream"].unique(), ftab["ftype"].unique()
    ):
        # Select data. We cannot assume that different streams or probes having the same metadata.
        mask = (
            (ftab["probe"] == probe)
            & (ftab["stream"] == stream)
            & (ftab["ftype"] == ftype)
        )
        _ftab = ftab.loc[mask]

        # Get the number of samples in each file.
        # fileTimeSec is not precise enough. We must use fileSizeBytes.
        def _get_sample_info(_ftab):
            nFileChan = _ftab["nSavedChans"].astype("int")
            nFileSamp = _ftab["fileSizeBytes"].astype("int") / (2 * nFileChan)
            firstSample = _ftab["firstSample"].astype("int")
            return nFileSamp, firstSample

        try:
            nFileSamp, firstSample = _get_sample_info(_ftab)
        except ValueError:
            logger.warn(
                "File found with metadata that could not be cast as an integer. This likely indicates an incomplete .meta produced by a crash. Please repair."
            )
            logger.warn(
                "Attempting to proceed anyway. This is experimental and could affect tExperiment values. Proceed with caution."
            )
            _ftab.loc[:, "fileSizeBytes"] = _ftab["fileSizeBytes"].fillna(-1)
            _ftab.loc[:, "firstSample"] = _ftab["firstSample"].fillna(-1)
            nFileSamp, firstSample = _get_sample_info(_ftab)

        # Get the expected 'firstSample' of the next file in a continous recording.
        nextSample = nFileSamp + firstSample

        # Check these expected 'firstSample' values against actual 'firstSample' values.
        # This tells us whether any file is actually contiguous with the one preceeding it.
        # A tolerance is used, because even in continuous recordings, files can overlap or gap by a sample.
        sampleMismatch = firstSample.values[1:] - nextSample.values[:-1]
        isContinuation = np.abs(sampleMismatch) <= tol
        isContinuation = np.insert(
            isContinuation, 0, False
        )  # The first file is, by definition, not a continuation.

        sampleMismatch = np.insert(
            sampleMismatch, 0, 0
        )  # The first file comes with no expectations, and therefore no mismatch.

        # For each 'series' of continuous files, get the first datetime (i.e. `fileCreateTime`) in that series.
        ser_dt0 = _ftab["fileCreateTime"].copy()
        ser_dt0[isContinuation] = np.NaN
        ser_dt0 = ser_dt0.fillna(method="ffill")

        # For each 'series' of continuous files, use the first datetime to compute the timedelta to the start of the experiment.
        exp_dt0 = _ftab["fileCreateTime"].min()
        ser_exp_td = ser_dt0 - exp_dt0

        # For each file, get the number of seconds from the start of that series, NOT accounting for samples
        # collected before we started saving to disk.*
        # *(SGLX starts counting samples as soon as acquisition starts, even before data is written to disk)
        file_t0 = _ftab["firstSample"].astype("float") / _ftab["imSampRate"].astype(
            "float"
        )

        # Now account for samples collected before we started writing to disk.
        ser_t0 = file_t0.copy()
        ser_t0[isContinuation] = np.NaN
        ser_t0 = ser_t0.fillna(method="ffill")
        file_ser_t = (
            file_t0 - ser_t0
        )  # This is the number of seconds from the start of the series

        # Finally, combine the (super-precise) offset of each file within its continuous series, with the (less precise*) offset of each series from the start of the experiment.
        # *(If there is only one series --i.e. no crashes or gaps -- there is no loss of precision. Otherwise, tests indicate that this value is precise to within a few msec.)
        ftab.loc[mask, "tExperiment"] = file_ser_t + ser_exp_td.dt.total_seconds()
        ftab.loc[mask, "dtExperiment"] = exp_dt0 + pd.to_timedelta(
            ftab.loc[mask, "tExperiment"], "s"
        )
        ftab.loc[mask, "isContinuation"] = isContinuation
        ftab.loc[mask, "sampleDiff"] = sampleMismatch
    return ftab


def get_experiment_sessions(sessions, experiment):
    """Get the subset of sessions needed by an experiment.

    Parameters:
    -----------
    sessions: list of dict
        The YAML specification of sessions for this subject.
    experiment: dict
        The YAML specification of this experiment for this subject.

    Returns:
    --------
    list of dict
    """
    return [
        session
        for session in sessions
        if session["id"] in experiment["recording_session_ids"]
    ]


def get_experiment_files_table(sessions, experiment):
    """Get all SpikeGLX files belonging to a single experiment.

    Parameters:
    -----------
    sessions: list of dict
        The YAML specification of sessions for this subject.
    experiment: dict
        The YAML specification of this experiment for this subject.

    Returns:
    --------
    list of pathlib.Path
    """
    files = list(
        it.chain.from_iterable(
            get_session_files_from_multiple_locations(session)
            for session in get_experiment_sessions(sessions, experiment)
        )
    )
    return get_gate_dir_trigger_file_index(
        get_start_times_relative_to_experiment(
            sglx.filelist_to_frame(files)
        )
    )


def _get_subalias_files(files_df, start_file, end_file, subalias_idx=None):
    # If there is no subalias (aka a single subalias).
    if subalias_idx is None:
        subalias_idx = -1
    subalias_df = files_df[
        parse_trigger_stem(start_file) : parse_trigger_stem(end_file)
    ].reset_index()
    subalias_df["subalias_idx"] = subalias_idx
    return subalias_df


def get_alias_files_table(sessions, experiment, alias):
    """Get all SpikeGLX files belonging to a single alias.

    Parameters:
    -----------
    sessions: list of dict
        The YAML specification of sessions for this subject.
    experiment: dict
        The YAML specification of this experiment for this subject.
    alias: list
        The YAML specification of this alias, for this experiment, for this subject.
        The following formats is expected:
            [
                {
                    'start_file': <start_file_stem>
                    'end_file': <end_file_stem>
                }, # "subalias" 0
                {
                    'start_file': <start_file_stem>
                    'end_file': <end_file_stem>
                }, # "subalias" 1
                ...
            ]
        The index of the subalias each file is taken from is specified in the 'subalias_idx' column in the
        returned frame. If there is a unique subalias, subalias_idx is set to -1

    Returns:
    --------
    pd.DataFrame:
        All files in each of the sub-aliases, in sorted order, inclusive of both start_file and end_file.
    """
    df = get_experiment_files_table(sessions, experiment)
    df = (
        sglx.set_index(df).reset_index(level=0).sort_index()
    )  # Make df sliceable using (run, gate, trigger)

    if (
        isinstance(alias, list) and len(alias) == 1
    ):  # Single subalias ("subalias_idx" is set to -1)
        return _get_subalias_files(
            df, alias[0]["start_file"], alias[0]["end_file"], subalias_idx=None
        )

    elif isinstance(alias, list):  # Multiple subaliases.
        return pd.concat(
            [
                _get_subalias_files(
                    df, subalias["start_file"], subalias["end_file"], subalias_idx=i
                )
                for i, subalias in enumerate(alias)
            ]
        ).reset_index()

    else:
        raise ValueError("Unrecognized format for alias:\n {alias}")
