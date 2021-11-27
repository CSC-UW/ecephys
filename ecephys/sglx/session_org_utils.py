"""
These functions assume session-style experiment specification and organization of SpikeGLX data.

Functions beginning with underscores generally return lists and dictionaries,
while functions without are generally wrappers that return DataFrames.

Session style organization looks like this:
- subject_dir/ (e.g. ANPIX11-Adrian/)
    - session_dir/ (e.g. 8-27-2021/)
        - SpikeGLX/ (aka "session_sglx_dir")
            - gate_dir/ (example: 8-27-2021_g0/)
            ...
            - gate_dir/ (example: 8-27-2021_Z_g0)


Later, it might make sense to add a layer to this hierarchy, such that it goes
subject_dir > session_dir > SpikeGLX > run_dir > gate_dir.
"""
import re
from itertools import chain
from pathlib import Path
import pandas as pd

from .file_mgmt import (
    filter_files,
    validate_sglx_path,
    get_gate_files,
    filelist_to_frame,
    loc,
    set_index,
)


def _get_session_style_path_parts(path):
    gate_dir, probe_dirname, fname = validate_sglx_path(path)
    session_sglx_dir = gate_dir.parent
    session_dir = session_sglx_dir.parent
    subject_dir = session_dir.parent
    return (
        subject_dir,
        session_dir.name,
        session_sglx_dir.name,
        gate_dir.name,
        probe_dirname,
        fname,
    )


def _get_gate_directories(session_sglx_dir):
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


def _get_session_files_from_single_location(session_sglx_dir):
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
            for gate_dir in _get_gate_directories(session_sglx_dir)
        )
    )

def get_session_files_from_single_location(session_sglx_dir):
    return filelist_to_frame(_get_session_files_from_single_location(session_sglx_dir))

def _get_session_files_from_multiple_locations(session):
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
    ap_files = filter_files(_get_session_files_from_single_location(Path(session["ap"])), stream="ap")
    lf_files = filter_files(_get_session_files_from_single_location(Path(session["lf"])), stream="lf")
    return ap_files + lf_files

def get_session_files_from_multiple_locations(session):
    return filelist_to_frame(_get_session_files_from_multiple_locations(session))

def _get_subject_files(sessions):
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
            _get_session_files_from_multiple_locations(session)
            for session in sessions
        )
    )


def get_subject_files(sessions):
    return filelist_to_frame(_get_subject_files(sessions))


def _get_sessions_yamlstream_files(stream):
    """Get SpikeGLX files of belonging to all YAML documents in a YAML stream.

    Parameters:
    -----------
    stream: list of dict
        The YAML specifications for each subject.

    Returns:
    --------
    dict of list of pathlib.Path, keyed by subject.
    """
    return {doc["subject"]: _get_subject_files(doc["recording_sessions"]) for doc in stream}


def get_yamlstream_files(stream):
    d = {doc["subject"]: get_subject_files(doc["recording_sessions"]) for doc in stream}
    return pd.concat(d.values(), keys=d.keys(), names=["subject"], sort=True)

##### Functions up to here deal with sessions.yaml, not experiments_and_aliases.yaml

def _get_experiment_sessions(sessions, experiment):
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
    return [session for session in sessions if session["id"] in experiment["recording_session_ids"]]

def _get_experiment_files(sessions, experiment):
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
    return list(
        chain.from_iterable(
            _get_session_files_from_multiple_locations(session)
            for session in _get_experiment_sessions(sessions, experiment)
        )
    )


def get_experiment_files(sessions, experiment):
    return filelist_to_frame(_get_experiment_files(sessions, experiment))


# TODO: This function seems like it belongs in file_mgmt.py, since it does not
# rely on sessions.yaml or experiments_and_aliases.yaml
def parse_trigger_stem(stem):
    x = re.search(r"_g\d+_t\d+\Z", stem)  # \Z forces match at string end.
    run = stem[: x.span()[0]]  # The run name is everything before the match
    gate = re.search(r"g\d+", x.group()).group()
    trigger = re.search(r"t\d+", x.group()).group()

    return (run, gate, trigger)


def get_alias_files(sessions, experiment, alias):
    """Get all SpikeGLX files belonging to a single alias.

    Parameters:
    -----------
    sessions: list of dict
        The YAML specification of sessions for this subject.
    experiment: dict
        The YAML specification of this experiment for this subject.
    alias: dict
        The YAML specification of this alias, for this experiment, for this subject.

    Returns:
    --------
    pd.DataFrame:
        All files in the alias, in sorted order, inclusive of both start_file and end_file.
    """
    df = get_experiment_files(sessions, experiment)
    df = (
        set_index(df).reset_index(level=0).sort_index()
    )  # Make df sliceable using (run, gate, trigger)
    return df[
        parse_trigger_stem(alias["start_file"]) : parse_trigger_stem(alias["end_file"])
    ].reset_index()

# TODO: This function could go with the sessions.yaml functions and be imported by the
# experiments_and_aliases.yaml functions.
def get_subject_document(yaml_stream, subject_name):
    """Get a subject's YAML document from a YAML stream."""
    matches = [doc for doc in yaml_stream if doc["subject"] == subject_name]
    assert len(matches) == 1, f"Exactly 1 YAML document should match {subject_name}"
    return matches[0]

def get_files(sessions_stream, experiments_stream, subject_name, experiment_name, alias_name=None, **kwargs):
    """Get all SpikeGLX files matching selection criteria."""
    sessions_doc = get_subject_document(sessions_stream, subject_name)
    experiments_doc = get_subject_document(experiments_stream, subject_name)

    sessions = sessions_doc['recording_sessions']
    experiment = experiments_doc['experiments'][experiment_name]

    df = (
        get_alias_files(sessions, experiment, experiment['aliases'][alias_name])
        if alias_name
        else get_experiment_files(sessions, experiment)
    )
    return loc(df, **kwargs)
