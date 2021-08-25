"""These functions assume session-style experiment specification and organization of SpikeGLX data.
This system would be very easy to extend so that each recording session can be on e.g. a different drive,
by simply moving the raw-data-root field into each session (and probably renaming it).

Functions beginning with underscores generally return lists and dictionaries,
while functions without are generally wrappers that return DataFrames. """

import re
from itertools import chain
from pathlib import Path
import pandas as pd

from .file_mgmt import (
    get_gate_files,
    filelist_to_frame,
    loc,
    parse_trigger_stem,
    set_index,
)


def _get_gate_directories(session_dir, run_name):
    """Get all gate directories belonging to a single run.

    Parameters:
    -----------
    session_dir: pathlib.Path
    run_name: string

    Returns:
    --------
    list of pathlib.Path
    """
    matches = [
        p
        for p in session_dir.glob(f"{run_name}_g*")
        if (p.is_dir() and re.search(r"_g\d+\Z", p.name))
    ]
    return sorted(matches)


def _get_run_files(session_dir, run_name):
    """Get all SpikeGLX files belonging to a single run.

    Parameters:
    -----------
    session_dir: pathlib.Path
    run_name: string

    Returns:
    --------
    list of pathlib.Path
    """
    return list(
        chain.from_iterable(
            get_gate_files(gate_dir)
            for gate_dir in _get_gate_directories(session_dir, run_name)
        )
    )


def get_run_files(session_dir, run_name):
    return filelist_to_frame(_get_run_files(session_dir, run_name))


def _get_session_files(root_dir, session):
    """Get all SpikeGLX files belonging to a single session.

    Parameters:
    -----------
    root_dir: pathlib.Path
        The path to the root raw data directory.
    session: dict
        The YAML specification for this session.

    Returns:
    --------
    list of pathlib.Path
    """
    return list(
        chain.from_iterable(
            _get_run_files(root_dir / session["directory"], run_name)
            for run_name in session["SpikeGLX-runs"]
        )
    )


def get_session_files(root_dir, session):
    return filelist_to_frame(_get_session_files(root_dir, session))


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
            _get_session_files(Path(doc["raw-data-root"]), session)
            for session in doc["recording-sessions"]
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
            _get_session_files(Path(doc["raw-data-root"]), session)
            for session in doc["experiments"][experiment_name]["recording-sessions"]
        )
    )


def get_experiment_files(doc, experiment_name):
    return filelist_to_frame(_get_experiment_files(doc, experiment_name))


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
    df = set_index(df).reset_index(
        level=0
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
