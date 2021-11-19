"""
These functions assume session-style experiment specification and organization of SpikeGLX data.
This system would be very easy to extend so that each recording session can be on e.g. a different drive,
by simply moving the raw-data-root field into each session (and probably renaming it).

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


def _get_session_files(session_sglx_dir):
    """Get all SpikeGLX files belonging to a single session.

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


def get_session_files(session_sglx_dir):
    return filelist_to_frame(_get_session_files(session_sglx_dir))


def _get_document_files(doc):
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
            _get_session_files(Path(doc["raw_data_directory"], session, "SpikeGLX"))
            for session in doc["recording_sessions"]
        )
    )


def get_document_files(doc):
    return filelist_to_frame(_get_document_files(doc))


def _get_subject_files(doc):
    return _get_document_files(doc)


def get_subject_files(doc):
    return get_document_files(doc)


def _get_yamlstream_files(stream):
    """Get SpikeGLX files of belonging to all YAML documents in a YAML stream.

    Parameters:
    -----------
    docs: list of dict
        The YAML specifications for each subject.

    Returns:
    --------
    list of pathlib.Path
    """
    return {doc["subject"]: _get_document_files(doc) for doc in stream}


def get_yamlstream_files(stream):
    d = {doc["subject"]: get_document_files(doc) for doc in stream}
    return pd.concat(d.values(), keys=d.keys(), names=["subject"], sort=True)


def _get_experiment_files(doc, experiment_name):
    """Get all SpikeGLX files belonging to a single experiment.

    Parameters:
    -----------
    doc: dict
        The YAML specification for this subject.
    experiment_name: string

    Returns:
    --------
    list of pathlib.Path
    """
    return list(
        chain.from_iterable(
            _get_session_files(Path(doc["raw_data_directory"], session, "SpikeGLX"))
            for session in doc["experiments"][experiment_name]["recording_sessions"]
        )
    )


def get_experiment_files(doc, experiment_name):
    return filelist_to_frame(_get_experiment_files(doc, experiment_name))


def parse_trigger_stem(stem):
    x = re.search(r"_g\d+_t\d+\Z", stem)  # \Z forces match at string end.
    run = stem[: x.span()[0]]  # The run name is everything before the match
    gate = re.search(r"g\d+", x.group()).group()
    trigger = re.search(r"t\d+", x.group()).group()

    return (run, gate, trigger)


def get_alias_files(doc, experiment_name, alias_name):
    """Get all SpikeGLX files belonging to a single alias.

    Parameters:
    -----------
    doc: dict
        The YAML specification for this subject.
    experiment_name: string
    alias_name: string

    Returns:
    --------
    pd.DataFrame:
        All files in the alias, in sorted order, inclusive of both start_file and end_file.
    """
    alias = doc["experiments"][experiment_name]["aliases"][alias_name]
    df = get_experiment_files(doc, experiment_name)
    df = (
        set_index(df).reset_index(level=0).sort_index()
    )  # Make df sliceable using (run, gate, trigger)
    return df[
        parse_trigger_stem(alias["start_file"]) : parse_trigger_stem(alias["end_file"])
    ].reset_index()


def get_subject_document(yaml_stream, subject_name):
    """Get a subject's YAML document from a YAML stream."""
    matches = [doc for doc in yaml_stream if doc["subject"] == subject_name]
    assert len(matches) == 1, f"Exactly 1 YAML document should match {subject_name}"
    return matches[0]


def get_files(yaml_stream, subject, experiment, alias=None, **kwargs):
    """Get all SpikeGLX files matching selection criteria."""
    doc = get_subject_document(yaml_stream, subject)
    df = (
        get_alias_files(doc, experiment, alias)
        if alias
        else get_experiment_files(doc, experiment)
    )
    return loc(df, **kwargs)
