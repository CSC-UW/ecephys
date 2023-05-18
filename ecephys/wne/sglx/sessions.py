"""
These functions resolve paths to SpikeGLX data, assuming that the data are
organized 'session-style' and described using the sglx_sessions.yaml format.

Session style organization looks like this:
- subject_dir/ (e.g. ANPIX11-Adrian/)*
    - session_dir/ (e.g. 8-27-2021/)
        - SpikeGLX/ (aka "session_sglx_dir")
            - gate_dir/ (example: 8-27-2021_g0/)
            ...
            - gate_dir/ (example: 8-27-2021_Z_g0)
              - Folder-per-probe organization (see SpikeGLX docs)

* A subject directory is not strictly necessary, but is typical.

The sglx_sessions.yaml consists of several 'YAML documents' (collectively
called a 'YAML stream'), each of which describes whee the data for one
subject can be found. One of these YAML documents might look like this:

---
subject: Adrian
recording_sessions:
  - id: 8-27-2021
    ap: /Volumes/neuropixel_archive/Data/chronic/CNPIX11-Adrian/8-27-2021/SpikeGLX/
    lf: /Volumes/NeuropixelNAS2/data1/CNPIX11-Adrian/8-27-2021/SpikeGLX/
  ...
  - id: 8-31-2021
    ap: /Volumes/neuropixel_archive/Data/chronic/CNPIX11-Adrian/8-31-2021/SpikeGLX/
    lf: /Volumes/NeuropixelNAS1/data/CNPIX11-Adrian/8-31-2021/SpikeGLX/
...

Note that the AP and LFP data, as well as data from different sessions, can be distributed
across different locations (e.g. different NAS devices). This is because the the sheer volume
of AP data often requires specialized storage.
"""
# TODO: It would probably be better to define a location priority list, rather than to
#       explicitly define AP and LF data locations as we do currently. This could be less
#       verbose and also allow splitting of data across locations based on factors other
#       than stream type.

from itertools import chain
import logging
from pathlib import Path
import re

import pandas as pd

from ecephys.sglx import file_mgmt

logger = logging.getLogger(__name__)


def get_gate_directories(session_sglx_dir):
    """Get all gate directories belonging to a single session.

    Parameters:
    -----------
    session_sglx_dir: pathlib.Path

    Returns:
    --------
    list of pathlib.Path
    """
    matches = [
        p
        for p in session_sglx_dir.glob(f"*_g*")
        if (p.is_dir() and re.search(r"_g\d+\Z", p.name))
    ]
    return sorted(matches)


def get_session_files_from_single_location(session_sglx_dir):
    """Get all SpikeGLX files belonging to a single session directory.

    Parameters:
    -----------
    session_sglx_dir: pathlib.Path

    Returns:
    --------
    list of pathlib.Path
    """
    return list(
        chain.from_iterable(
            file_mgmt.get_gate_files(gate_dir)
            for gate_dir in get_gate_directories(session_sglx_dir)
        )
    )


def get_session_files_from_multiple_locations(session):
    """Get all SpikeGLX files belonging to a single session.
    The AP and LF files may be stored in separate locations.

    Parameters:
    -----------
    session: dict
        From sessions.yaml, must have fields 'ap' and 'lf' pointing to respective data locations.

    Returns:
    --------
    list of pathlib.Path
    """
    ap_files = file_mgmt.filter_files(
        get_session_files_from_single_location(Path(session["ap"])), stream="ap"
    )
    if not ap_files:
        logger.warning(
            f"No AP files found in directory: {session['ap']}. Do you need to update this subject's YAML file?"
        )
    lf_files = file_mgmt.filter_files(
        get_session_files_from_single_location(Path(session["lf"])), stream="lf"
    )
    if not lf_files:
        logger.warning(
            f"No LF files found in directory: {session['lf']}. Do you need to update this subject's YAML file?"
        )
    return ap_files + lf_files


def get_session_style_path_parts(fpath):
    """Get all elements of a session-style filepath.

    Parameters:
    -----------
    fpath: pathlib.Path
        Must be a file (e.g. run0_g0_t0.lf.bin), not a directory.

    Returns:
    --------
    list of pathlib.Path or str containing the following elements::
        - Root directory  (eg path/to/my/project)
        - Subject directory name (eg 'CNPIX11-Adrian')
        - Session directory name (eg '8-27-2021')
        - Session SGLX directory name (ie 'SpikeGLX')
        - Gate directory name
        - Probe directory name
        - Filename
    """
    gate_dir, probe_dirname, fname = file_mgmt.validate_sglx_path(fpath)
    session_sglx_dir = gate_dir.parent
    session_dir = session_sglx_dir.parent
    subject_dir = session_dir.parent
    root_dir = subject_dir.parent
    return (
        root_dir,
        subject_dir.name,
        session_dir.name,
        session_sglx_dir.name,
        gate_dir.name,
        probe_dirname,
        fname,
    )


def get_filepath_relative_to_session_directory_parent(path):
    """Get only the path parts after and including a
    session directory, discarding path parts that precede
    the session directory.

    In other words, given a path to a SpikeGLX file under
    session-style organization, return only the path parts
    which fall under the purview of that specification.

    Parameters:
    -----------
    path: pathlib.Path
        Must be a file (e.g. run0_g0_t0.lf.bin), not a directory.

    Returns:
    --------
    pathlib.Path
    """
    gate_dir, probe_dirname, fname = file_mgmt.validate_sglx_path(path)
    session_sglx_dir = gate_dir.parent
    session_dir = session_sglx_dir.parent
    return path.relative_to(session_dir.parent)


def mirror_raw_data_path(mirror_parent, path):
    """Mirror a path to raw SpikeGLX data, maintaining session-style data/directory organization, but at a new path root.
    For example...
    `/foo/bar/1-1-2021/SpikeGLX/1-1-2021_g0/1-1-2021_g0_imec0/1-1-2021_g0_t0.imec0.lf.bin`
    ...might become...
    `/baz/qux/1-1-2021/SpikeGLX/1-1-2021_g0/1-1-2021_g0_imec0/1-1-2021_g0_t0.imec0.lf.bin`

    Parameters:
    -----------
    mirror_parent: pathlib.Path
        The new root to mirror at, e.g. `/baz/qux`
    path: pathlib.Path
        The old filepath, e.g. `/foo/bar/1-1-2021/SpikeGLX/1-1-2021_g0/1-1-2021_g0_imec0/1-1-2021_g0_t0.imec0.lf.bin`

    Returns:
    --------
    pathlib.Path: the re-rooted path, e.g. `/baz/qux/1-1-2021/SpikeGLX/1-1-2021_g0/1-1-2021_g0_imec0/1-1-2021_g0_t0.imec0.lf.bin`
    """
    return mirror_parent / get_filepath_relative_to_session_directory_parent(path)


def mirror_raw_data_paths(mirror_parent, paths):
    return [mirror_raw_data_path(mirror_parent, p) for p in paths]


def _parse_trigger_stem(stem):
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


def slice_by_trigger_stem(
    df: pd.DataFrame, start_stem: str, end_stem: str
) -> pd.DataFrame:
    df = df.copy()
    # Make df sliceable using (run, gate, trigger)
    df = file_mgmt.set_index(df).reset_index(level=0).sort_index()
    # Select desired files
    start_rgt = _parse_trigger_stem(start_stem)
    end_rgt = _parse_trigger_stem(end_stem)
    df = df[start_rgt:end_rgt]
    return df.reset_index()


def slice_by_datetime(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)

    fileStartDatetime = df["sessionPrbAcqDatetime"]
    fileEndDatetime = fileStartDatetime + pd.to_timedelta(df["fileDuration"], "s")
    mask = (
        (
            (start <= fileStartDatetime) & (end >= fileEndDatetime)
        )  # Subalias starts before file and ends after it, OR...
        | (
            (end >= fileStartDatetime) & (end <= fileEndDatetime)
        )  # Subalias ends during file, OR...
        | (  #
            (start >= fileStartDatetime) & (start <= fileEndDatetime)
        )  # Subalias starts during file
    )
    return df.loc[mask].reset_index(drop=True)


# TODO: There is really no concept of a subalias anymore. Even the alias concept should probably be retired.
def get_subalias_frame(sessionFrame: pd.DataFrame, subalias: dict):
    if ("start_file" in subalias) and ("end_file" in subalias):
        return slice_by_trigger_stem(
            sessionFrame, subalias["start_file"], subalias["end_file"]
        )
    if ("start_time" in subalias) and ("end_time" in subalias):
        return slice_by_datetime(
            sessionFrame, subalias["start_time"], subalias["end_time"]
        )
