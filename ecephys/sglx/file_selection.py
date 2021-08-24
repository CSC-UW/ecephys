"""These functions assume session-style experiment specification and organization of SpikeGLX data.
This system would be very easy to extend so that each recording session can be on e.g. a different drive,
by simply moving the raw-data-root field into each session (and probably renaming it)."""

import re
from itertools import chain
from pathlib import Path
import pandas as pd
from pandas.api.types import CategoricalDtype

from .paths import get_gate_files, parse_sglx_fname


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


def _sort_strings_by_integer_suffix(strings):
    """Sort strings such that foo2 < foo10, contrary to default lexical sorting."""
    return sorted(strings, key=lambda string: int(re.split(r"(^[^\d]+)", string)[-1]))


def filelist_to_frame(files, run_order="infer"):
    """Create a DataFrame that allows for easy sorting, slicing, selection, etc. of
    files.

    Parameters:
    -----------
    files: list of pathlib.Path, must be SGLX files
    run_order: string, list, or None
        If `infer` (default), assume that the order of appearrance of runs in the filelist reflects
        the actual (chronological) ordering of runs. This should be true if the data were loaded using
        any of the functions in this module or in ecephys.sglx.paths.
        If `list`, use the run ordering provided in the list. List items must be unique.
        If `None`, simple lexical sorting of run names will be used when sorting and selecting data.

    Returns:
    --------
    pd.DataFrame with columns [run, gate, trigger, probe, stream, ftype, path].
    """
    runs, gates, triggers, probes, streams, ftypes = zip(
        *[parse_sglx_fname(f.name) for f in files]
    )
    df = pd.DataFrame(
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

    # Now use categorical types where appropriate, to allow for proper sorting
    # that respects e.g. the order of runs in the yaml spec.
    if run_order == "infer":
        run_dtype = CategoricalDtype(df["run"].unique(), ordered=True)
    elif run_order:
        run_dtype = CategoricalDtype(run_order, ordered=True)
    else:
        run_dtype = CategoricalDtype(df["run"].unique(), ordered=False)

    df["run"] = df["run"].astype(run_dtype)

    # Make sure that e.g. g2 < g10
    for x in ["gate", "trigger", "probe"]:
        df[x] = df[x].astype(
            CategoricalDtype(
                _sort_strings_by_integer_suffix(df[x].unique()), ordered=True
            )
        )

    # Stream type is unordered.
    df["stream"] = df["stream"].astype(
        CategoricalDtype(df["stream"].unique(), ordered=False)
    )

    return df.set_index(
        ["run", "gate", "trigger", "probe", "stream", "ftype"]
    ).sort_index()


def get_run_files(session_dir, run_name):
    """Get all SpikeGLX files belonging to a single run.

    Parameters:
    -----------
    session_dir: pathlib.Path
    run_name: string

    Returns:
    --------
    pd.DataFrame
    """
    return filelist_to_frame(_get_run_files(session_dir, run_name))


def get_session_files(root_dir, session):
    """Get all SpikeGLX files belonging to a single session.

    Parameters:
    -----------
    root_dir: pathlib.Path
        The path to the root raw data directory.
    session: dict
        The YAML specification for this session.

    Returns:
    --------
    pd.DataFrame
    """
    return filelist_to_frame(_get_session_files(root_dir, session))


def get_document_files(doc):
    """Get all SpikeGLX files belonging to a single subject's YAML document.

    Parameters:
    -----------
    doc: dict
        The YAML specification for this subject.

    Returns:
    --------
    pd.DataFrame
    """
    return filelist_to_frame(_get_document_files(doc))


def get_experiment_files(doc, experiment_name):
    """Get all SpikeGLX files belonging to a single experiment.

    Parameters:
    -----------
    doc: dict
        The YAML specification for this subject.
    experiment_name: string

    Returns:
    --------
    pd.DataFrame
    """
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
    return df[
        parse_trigger_stem(alias["start_file"]) : parse_trigger_stem(alias["end_file"])
    ]


def _slice_files_by_name(files, start, end):
    """Files must be sorted BY STEM (e.g. separated by probe) before using this function,
    else pd.Index.slice_locs will correctly raise an error."""
    parses = [parse_sglx_fname(f.name) for f in files]
    stems = [
        f"{run}_{gate}_{trigger}" for run, gate, trigger, probe, stream, ftype in parses
    ]
    (start, end) = pd.Index(stems).slice_locs(start, end)
    return files[start:end]


def _get_alias_files(doc, experiment_name, alias_name):
    alias = doc["experiments"][experiment_name]["aliases"][alias_name]
    experiment_files = get_experiment_files(doc, experiment_name)
    alias_files_by_probe = {
        probe: _slice_files_by_name(files, alias["start_file"], alias["end_file"])
        for probe, files in separate_files_by_probe(experiment_files).items()
    }
    return [
        f
        for f in experiment_files
        if f in list(chain.from_iterable(alias_files_by_probe.values()))
    ]
