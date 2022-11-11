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
import re
import logging
from itertools import chain
from pathlib import Path

from ecephys.sglx.file_mgmt import (
    filter_files,
    validate_sglx_path,
    get_gate_files,
)

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
            get_gate_files(gate_dir)
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
    ap_files = filter_files(
        get_session_files_from_single_location(Path(session["ap"])), stream="ap"
    )
    if not ap_files:
        logger.warning(
            f"No AP files found in directory: {session['ap']}. Do you need to update this subject's YAML file?"
        )
    lf_files = filter_files(
        get_session_files_from_single_location(Path(session["lf"])), stream="lf"
    )
    if not lf_files:
        logger.warning(
            f"No LF files found in directory: {session['lf']}. Do you need to update this subject's YAML file?"
        )
    return ap_files + lf_files


# TODO: Deprecated. Remove?
def get_subject_files(sessions):
    """Get all SpikeGLX files belonging to a single subject's YAML document.

    Parameters:
    -----------
    doc: dict
        The YAML specification for this subject.

    Returns:
    --------
    list of pathlib.Path
    """
    return list(
        chain.from_iterable(
            get_session_files_from_multiple_locations(session) for session in sessions
        )
    )


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
    gate_dir, probe_dirname, fname = validate_sglx_path(fpath)
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
    gate_dir, probe_dirname, fname = validate_sglx_path(path)
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
