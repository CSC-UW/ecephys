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
from itertools import chain

# Non-core imports
import pandas as pd
from pandas.api.types import CategoricalDtype
from .external.readSGLX import readMeta


def parse_sglx_fname(fname):
    """Parse recording identifiers from a SpikeGLX style filename stem.

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


###############################################################################
# The following functions require pandas and other non-core Python packages
##############################################################################


def read_metadata(files):
    """Takes a list of pathlib.Path"""
    meta_dict = [readMeta(f) for f in files]
    return pd.DataFrame(meta_dict).assign(path=files)


def _filelist_to_frame(files):
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
    meta_df = (
        read_metadata(files)
        .astype({"fileCreateTime": "datetime64[ns]"})
        .sort_values("fileCreateTime", ascending=True)
        .reset_index(drop=True)
    )
    files_df = _filelist_to_frame(files)
    return meta_df.merge(files_df, on="path")


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
    return df.loc[(df[list(kwargs)] == pd.Series(kwargs)).all(axis=1)]
