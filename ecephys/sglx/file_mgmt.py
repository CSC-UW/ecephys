"""Most functions in this module assume folder-per-probe organization:
- run_dir/ (example: 3-1-2021/)
  - gate_dir_g0/ (example: 3-1-2021_g0/)
  ...
  - gate_dir_gN/
    - probe_dir_imec0/ (example: 3-1-2021_g0_imec0/)
    ...
    - probe_dir_imecN/
      - trigger_file_gN_imecN_t0.lf.bin
      - trigger_file_gN_imecN_t0.lf.meta
      - trigger_file_gN_imecN_t0.ap.bin
      - trigger_file_gN_imecN_t0.ap.meta
      ...
      - trigger_file_gN_imecN_tN.lf.bin
      - trigger_file_gN_imecN_tN.lf.meta
      - trigger_file_gN_imecN_tN.ap.bin
      - trigger_file_gN_imecN_tN.ap.meta

Technically, SGLX tools do not define/enforce that all gates belonging to a
run reside in the same run directory (i.e. there is no such thing as a run_dir),
but many people organize their data this way, and supporting it is harmless.

However, if you only respect this hierarchy up to a certain level,
(e.g. you don't use a run_dir -- I don't), all the functions up to that level
will still work.

All functions that get/take a list of files/directories input/output lists
of pathlib.Path objects.

The logical ordering of files (defined by recursive traversal of the hierarchy)
is consistent across functions should always be preserved.

No functions in this file should require non-core packages, a config, or
depend on any organization schema beyond that of official SpikeGLX tools.
"""

import re
import ast
import logging
import pathlib
from itertools import chain

# Non-core imports
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from .external.readSGLX import readMeta

logger = logging.getLogger(__name__)


def parse_sglx_fname(fname):
    """Parse recording identifiers from a SpikeGLX style filename.

    Parameters
    ---------
    fname: str
        The filename to parse, e.g. "my-run-name_g0_t1.imec2.lf.bin"

    Returns
    -------
    run: str
        The run name, e.g. "my-run-name".
    gate: str
        The gate identifier, e.g. "g0".
    trigger: str
        The trigger identifier, e.g. "t1".
    probe: str
        The probe identifier, e.g. "imec2"
    stream: str
        The data type identifier, "lf" or "ap"
    ftype: str
        The file type identifier, "bin" or "meta"

    Examples
    --------
    >>> parse_sglx_fname('3-1-2021_A_g1_t0.imec0.lf.meta')
    ('3-1-2021_A', 'g1', 't0', 'imec0', 'lf', 'meta')
    """
    x = re.search(
        r"_g\d+_t\d+\.imec\d+.(ap|lf).(bin|meta)\Z", fname
    )  # \Z forces match at string end.
    run = fname[: x.span()[0]]  # The run name is everything before the match
    gate = re.search(r"g\d+", x.group()).group()
    trigger = re.search(r"t\d+", x.group()).group()
    probe = re.search(r"imec\d+", x.group()).group()
    stream = re.search(r"(ap|lf)", x.group()).group()
    ftype = re.search(r"(bin|meta)", x.group()).group()

    return (run, gate, trigger, probe, stream, ftype)


def _sort_strings_by_integer_suffix(strings):
    """Sort strings such that foo2 < foo10, contrary to default lexical sorting."""
    return sorted(strings, key=lambda string: int(re.split(r"(^[^\d]+)", string)[-1]))


def get_trigger_files(probe_dir):
    """Get all SGLX files in a probe directory.

    Parameters:
    -----------
    probe_dir: pathlib.Path
        Path to SGLX probe directory.

    Returns
    -------
    list of pathlib.Path
        All trigger files in the directory, regardless of lf/ap/bin/meta type,
        with trigger order preserved.
    """
    matches = [
        p
        for p in probe_dir.glob("*_g*_t*.imec[0-9].*.*")
        if (
            p.is_file()
            and re.search(r"_g\d+_t\d+\.imec\d+.(ap|lf).(bin|meta)\Z", p.name)
        )
    ]
    return sorted(matches)


# This function is probably extraneous.
def _get_unique_trigger_stems(probe_dir):
    """Get all unique trigger stems in a probe directory.

    Parameters:
    -----------
    probe_dir: pathlib.Path
        Path to SGLX probe directory.

    Returns
    -------
    list of pathlib.Path
        All unique stems in sorted order.

    Examples:
    ---------
    >>> probe_dir = Path('path/to/3-1-2021_A_g1_imec0')
    >>> get_unique_trigger_stems(probe_dir):
    ['3-1-2021_A_g1_t0', '3-1-2021_A_g1_t1']
    """
    parses = [parse_sglx_fname(f.name) for f in get_trigger_files(probe_dir)]
    stems = [
        f"{run}_{gate}_{trigger}" for run, gate, trigger, probe, stream, ftype in parses
    ]
    return sorted(dict.fromkeys(stems))


def get_probe_directories(gate_dir):
    """Get all probe directories in a gate directory.

    Parameters:
    -----------
    gate_dir: pathlib.Path
        Path to SGLX gate directory.

    Returns
    -------
    list of pathlib.Path
        All matching probe directories in sorted order.
    """
    matches = [
        p
        for p in gate_dir.glob(f"{gate_dir.name}_imec[0-9]")
        if (p.is_dir() and re.search(r"_g\d+_imec\d+\Z", p.name))
    ]
    return sorted(matches)


def get_gate_files(gate_dir):
    """Get all SGLX files in a gate directory.

    Parameters:
    -----------
    gate_dir: pathlib.Path
        Path to SGLX gate directory.

    Returns
    -------
    list of pathlib.Path
        All files in sorted order.
    """
    return list(
        chain.from_iterable(
            get_trigger_files(probe_dir)
            for probe_dir in get_probe_directories(gate_dir)
        )
    )


def get_gate_directories(run_dir):
    """Get all gate directories in a run directory.

    Technically, SGLX tools do not define/enforce that all gates belonging
    to a run reside in the same run directory, but many people organize
    their data this way.

    Parameters:
    -----------
    run_dir: pathlib.Path
        Path to SGLX run directory.

    Returns:
    --------
    list of pathlib.Path
        All gate directories in sorted order.
    """
    matches = [
        p
        for p in run_dir.glob(f"{run_dir.name}_g*")
        if (p.is_dir() and re.search(r"_g\d+\Z", p.name))
    ]
    return sorted(matches)


def get_run_files(run_dir):
    """Get all SGLX files in a run directory.

    Technically, SGLX tools do not define/enforce that all gates belonging
    to a run reside in the same run directory, but many people organize
    their data this way.

    Parameters:
    -----------
    run_dir: pathlib.Path
        Path to SGLX run directory.

    Returns:
    --------
    list of pathlib.Path
        All SGLX files in sorted order.
    """
    return list(
        chain.from_iterable(
            get_gate_files(gate_dir) for gate_dir in get_gate_directories(run_dir)
        )
    )


def filter_files(
    files, run=None, gate=None, trigger=None, probe=None, stream=None, ftype=None
):
    """Filter a list of SGLX files.

    Parameters:
    -----------
    files: list of pathlib.Path
        Must be SGLX files.
    run: string
        Optionally keep only files that match this run, e.g. '3-1-2021'.
    gate: string
        Optionally keep only files that match this gate, e.g. 'g0'.
    trigger: string
        Optionally keep only files that match this trigger, e.g. 't0'.
    probe: string
        Optionally keep only files that match this trigger, e.g. 'imec0'.
    stream: string
        Optionally keep only files that match this stream, e.g. 'lf'.
    ftype: string
        Optionally keep only files that match this filetype, e.g. 'bin'.

    Returns:
    --------
    list of pathlib.Path:
        All matching files, with their original order preserved.
    """
    desired = (run, gate, trigger, probe, stream, ftype)

    def keep_file(fname):
        actual = parse_sglx_fname(fname)
        keep = map(lambda x, y: x is None or x == y, desired, actual)
        return all(keep)

    return [f for f in files if keep_file(f.name)]


def remove_suffixes(
    files, regex=r"\.imec\d+\.(lf|ap)\.(bin|meta)", remove_duplicates=True
):
    """Remove suffixes from a list of files, to preserve only trigger stems
    and preceding path elements."""
    path_stems = [f.parent / re.sub(regex, "", f.name) for f in files]
    return list(dict.fromkeys(path_stems)) if remove_duplicates else path_stems


def separate_files_by_probe(files):
    """Return a dictionary, keyed by probe, with files separated by probe.
    Original order of files is preserved."""
    parses = [parse_sglx_fname(f.name) for f in files]
    probes = [probe for run, gate, trigger, probe, stream, ftype in parses]
    unique_probes = sorted(dict.fromkeys(probes))
    return {
        probe: [f for f, p in zip(files, probes) if p == probe]
        for probe in unique_probes
    }


def validate_sglx_path(path):
    """Check that file name and path obey SpikeGLX folder-per-probe naming
    conventions. Raise a ValuError if they do not, otherwise return the
    validated parts."""
    try:
        (run, gate, trigger, probe, stream, ftype) = parse_sglx_fname(path.name)
    except AttributeError:
        raise ValueError("Invalid file name.")

    try:
        probe_dir = path.parent
        gate_dir = probe_dir.parent
        assert probe_dir.name == f"{run}_{gate}_{probe}"
        assert gate_dir.name == f"{run}_{gate}"
    except AssertionError:
        raise ValueError("Invalid path.")

    return gate_dir, probe_dir.name, path.name


def replace_ftype(path, extension, remove_probe=False, remove_stream=False):
    """Replace a SpikeGLX filetype extension (i.e. .bin or .meta), and optionally strip
    the probe and/or stream suffixes (e.g. .imec0 and .lf) while doing so.

    Parameters:
    -----------
    path: pathlib.Path
    extension: str
        The desired final suffix(es), e.g. '.emg.nc' or '.txt'
    remove_probe: bool (default: False)
        If true, strip the probe suffix.
    remove_stream: bool (default=False)
        If True, strip the stream suffix.
    """
    run, gate, trigger, probe, stream, ftype = parse_sglx_fname(path.name)

    name = path.with_suffix(extension).name
    name = name.replace(f".{probe}", "") if remove_probe else name
    name = name.replace(f".{stream}", "") if remove_stream else name

    return path.with_name(name)


###############################################################################
# The following functions require pandas and other non-core Python packages
##############################################################################


def _try_casting(df, col, cast_to, fill_value):
    try:
        df = df.astype({col: cast_to})
    except ValueError:
        logger.warning(
            f"File found with {col} that could not be cast as {cast_to}. This likely indicates an incomplete .meta produced by a crash. Please repair."
        )
        for i, item in enumerate(df[col]):
            try:
                df[col].iloc[i : i + 1].astype(cast_to)
            except ValueError:
                logger.warning(f"Problematic file: {df['path'].iloc[i]}")
        logger.warning(
            "Attempting to proceed anyway. This is experimental and could affect downstream results. Proceed with caution."
        )
        df[col] = df[col].fillna(fill_value)
    return df


def read_metadata(files):
    """Takes a list of pathlib.Path
    See https://billkarsh.github.io/SpikeGLX/Sgl_help/Metadata_30.html"""
    meta_dict = [readMeta(f) for f in files]
    df = pd.DataFrame(meta_dict)

    metadata_is_missing = df.isna().all(
        axis=1
    )  # Sometimes a file exists but is empty, and no metadata is found
    df = df.assign(path=files)
    df = df[~metadata_is_missing]  # Drop the empty files
    if metadata_is_missing.any():  # Warn the user
        for f in df[metadata_is_missing].path.to_numpy():
            logging.info(
                f"No metadata found for {f}. SpikeGLX probably wrote an empty file. Dropping."
            )

    meta_types = dict()
    meta_types["always_present"] = {
        "appVersion": object,
        "fileCreateTime": "datetime64[ns]",
        "fileName": object,
        "fileSHA1": object,
        "fileSizeBytes": int,
        "fileTimeSecs": float,
        "firstSample": int,
        "gateMode": object,
        "nSavedChans": int,
        "snsSaveChanSubset": object,
        "syncSourceIdx": int,
        "syncSourcePeriod": float,
        "trigMode": object,
        "typeImEnabled": int,
        "typeNiEnabled": int,
        "typeThis": object,
        "userNotes": object,
        "snsShankMap": object,
        "snsChanMap": object,
        "path": object,
    }
    meta_types["if_using_imec"] = {
        "acqApLfSy": object,
        "imAiRangeMax": float,
        "imAiRangeMin": float,
        "imCalibrated": bool,
        "imDatApi": object,
        "imDatBs_fw": object,
        "imDatBsc_fw": object,
        "imDatBsc_hw": object,
        "imDatBsc_pn": object,
        "imDatBsc_sn": object,
        "imDatFx_hw": object,
        "imDatFx_pn": object,
        "imDatFx_sn": object,
        "imDatHs_fw": object,  # Not documented in SpikeGLX manual, but present.
        "imDatHs_hw": object,
        "imDatHs_pn": object,
        "imDatHs_sn": object,
        "imDatPrb_dock": int,
        "imDatPrb_pn": object,
        "imDatPrb_port": int,
        "imDatPrb_slot": int,
        "imDatPrb_sn": object,
        "imDatPrb_type": int,
        "imLEDEnable": bool,
        "imRoFile": object,
        "imSampRate": float,
        "imTrgRising": bool,
        "imTrgSource": int,
        "snsApLfSy": object,
        "syncImInputSlot": int,
        "imroTbl": object,
    }
    meta_types["if_using_timed_trigger"] = {
        "trgTimIsHInf": bool,
        "trgTimIsNInf": bool,
        "trgTimNH": float,
        "trgTimTH": float,
        "trgTimTL": float,
        "trgTimTL0": float,
    }
    meta_types["maybe_present"] = {
        "nDataDirs": int,
        "rmt_USERTYPE": object,
        "typeObEnabled": int,
        "imIsSvyRun": bool,
        "imMaxInt": int,
        "imStdby": object,
        "imSvyMaxBnk": object,
    }

    for group, types in meta_types.items():
        missing = set(types) - set(df.columns)
        if missing:
            logger.debug(f"Metadata fields {missing} from group {group} not found.")
            for field in missing:
                types.pop(field, None)
        try:
            df = df.astype(types)
        except ValueError:
            df = _try_casting(df, "fileSizeBytes", int, -1)
            df = _try_casting(df, "firstSample", int, -1)
            df = df.astype(types)

    extra = set(df.columns) - set.union(
        *(set(types) for group, types in meta_types.items())
    )
    if extra:
        print(f"Found unexpected metadata fields: {extra}")

    df["acqApLfSy"] = df["acqApLfSy"].apply(ast.literal_eval)
    df["snsApLfSy"] = df["snsApLfSy"].apply(ast.literal_eval)
    df["fileName"] = df["fileName"].apply(pathlib.Path)
    df["imRoFile"] = df["imRoFile"].apply(pathlib.Path)

    return df.sort_values("fileCreateTime", ascending=True).reset_index(drop=True)


def parse_sglx_fnames(files):
    """Return a list of files in as a dataframe for easy slicing, selecting, etc.

    Parameters:
    -----------
    files: list of pathlib.Path, must be SGLX files

    Returns:
    --------
    pd.DataFrame with columns [run, gate, trigger, probe, stream, ftype, path].
    """
    runs, gates, triggers, probes, streams, ftypes = zip(
        *[parse_sglx_fname(f.name) for f in files]
    )
    return pd.DataFrame(
        {
            "run": runs,
            "gate": gates,
            "trigger": triggers,
            "probe": probes,
            "stream": streams,
            "ftype": ftypes,
            "path": files,
        }
    )


def filelist_to_frame(files):
    """Return a list of files in as a dataframe for easy slicing, selecting, etc.
    Includes metadata, and orders files by fileCreateTime.

    Parameters:
    -----------
    files: list of pathlib.Path, must be SGLX files

    Returns:
    --------
    pd.DataFrame with columns [run, gate, trigger, probe, stream, ftype, path].
    """
    if not files:
        return pd.DataFrame()

    meta_df = read_metadata(files)
    files_df = parse_sglx_fnames(files)
    df = meta_df.merge(files_df, on="path")

    # Create nFileSamp column, since it is so useful
    df["nFileSamp"] = df["fileSizeBytes"] / (2 * df["nSavedChans"])
    # Last sample, measured from the start of that probe's acquisition, in that probe's samplebase.
    df["lastSample"] = df["firstSample"] + df["nFileSamp"] - 1
    df["fileDuration"] = (
        df["nFileSamp"] / df["imSampRate"]
    )  # More precise than fileTimeSecs

    # First timestamp, measured from the start of that probe's acquisition, in that probe's timebase.
    df["firstTime"] = df["firstSample"].astype("float") / df["imSampRate"]
    # Last timestamp, measured from the start of that probe's acquisition, in that probe's timebase.
    df["lastTime"] = df["lastSample"].astype("float") / df["imSampRate"]

    return df


def get_semicontinuous_segments(df: pd.DataFrame, tol=100):
    """Get semicontinuous rows of a file frame.
    We call these `semicontinuous`, because files can be overlapped or gapped.

    Parameters
    ==========
    tol: int
        The number of samples that continuous files are allowed to be overlapped or gapped by.
        This needs to be fairly high since, when recording with multiple probes, one will be nearly perfect (tol=1),
        while the other will be much sloppier -- perhaps due to the different sampling rates on the two probes.

    Returns
    =======
    acqs: list[pd.DataFrame]
        The input frame, sliced into blocks of semicontinuous files
    segments: list[pd.Series]
        Summary info about each semicontinuous segment
    """
    assert len(df["probe"].unique()) == 1, "Frame include exactly 1 probe."
    assert len(df["stream"].unique()) == 1, "Frame include exactly 1 stream."
    assert len(df["ftype"].unique()) == 1, "Frame include exactly 1 filetype."
    df = df.copy()

    # Get the expected 'firstSample' of the next file in a continous recording.
    nextSample = df["lastSample"] + 1

    # Check these expected 'firstSample' values against actual 'firstSample' values.
    # This tells us whether any file is actually contiguous with the one preceeding it.
    sampleMismatch = df["firstSample"].values[1:] - nextSample.values[:-1]
    sampleMismatch = np.insert(
        sampleMismatch, 0, 0
    )  # The first file comes with no expectations, and therefore no mismatch.

    # A tolerance is used, because even in continuous recordings, files can overlap or gap by a sample.
    isContinuation = np.abs(sampleMismatch) <= tol
    isContinuation[0] = False  # The first file is, by definition, not a continuation.

    # Group files by their acquisiton ID, so that all files in a semicontinious acquisition block have the same ID.
    df["acquisitionID"] = (~isContinuation).cumsum() - 1
    acqs = [
        df.loc[df["acquisitionID"] == id] for id in sorted(df["acquisitionID"].unique())
    ]

    # Get summary information about each segment
    segments = list()
    for acq in acqs:
        # Keep all metadata fields if their value is the same for every file in the segment
        seg = acq.loc[:, acq.nunique() == 1].iloc[0].copy()
        # Number of samples written to disk in this semicontinuous block
        seg["nPrbAcqSamples"] = acq["nFileSamp"].sum()
        # First sample, measured from the start of this probe's acquisition, in this probe's samplebase.
        seg["firstPrbAcqSample"] = acq["firstSample"].min()
        # Last sample, measured from the start of this probe's acquisition, in this probe's samplebase.
        seg["lastPrbAcqSample"] = acq["lastSample"].max()
        seg["nDuplicateSamples"] = seg["nPrbAcqSamples"] - (
            seg["lastPrbAcqSample"] - seg["firstPrbAcqSample"] + 1
        )  # Negative values imply dropped samples
        # Datetime start of the recording, NOT acquisition.
        seg["prbRecDatetime"] = acq["fileCreateTime"].min()
        # First timestamp, measured from the start of this probe's acquisition, in this probe's samplebase.
        seg["firstTime"] = acq["firstTime"].min()
        segments.append(seg)

    return acqs, segments


def set_index(df):
    """Set an index with intelligent sorting of identifier strings."""
    df.astype({"fileCreateTime": "datetime64[ns]"}).sort_values(
        "fileCreateTime", ascending=True
    ).reset_index(drop=True)

    # df is already sorted chronologically, so runs will appear in order.
    df["run"] = df["run"].astype(CategoricalDtype(df["run"].unique(), ordered=True))

    # Make sure that e.g. g2 < g10
    for x in ["gate", "trigger", "probe"]:
        df[x] = df[x].astype(
            CategoricalDtype(
                _sort_strings_by_integer_suffix(df[x].unique()), ordered=True
            )
        )

    return df.set_index(
        ["fileCreateTime", "run", "gate", "trigger", "probe", "stream", "ftype"]
    ).sort_index()


def xs(df, **kwargs):
    """Select rows of a MultiIndex DataFrame based on index labels.

    Examples:
    ---------
    >>> df = filelist_to_frame(files)
    >>> df = _set_index(df)
    >>> xs(df, stream='lf', ftype='bin')
    >>> df.pipe(xs, stream='lf', ftype='bin')
    """
    return (
        df.xs(tuple(kwargs.values()), level=tuple(kwargs.keys()), drop_level=False)
        if kwargs
        else df
    )


def loc(df, **kwargs):
    """Select rows of a DataFrame based on column labels.

    Examples:
    ---------
    >>> df = filelist_to_frame(files)
    >>> loc(df, stream='lf', ftype='bin')
    >>> df.pipe(loc, stream='lf', ftype='bin')
    """
    return df.loc[(df[list(kwargs)] == pd.Series(kwargs, dtype="object")).all(axis=1)]
