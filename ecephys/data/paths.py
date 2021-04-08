import re
import yaml
from pathlib import Path
import pandas as pd
from ..utils import flatten


def get_datapath_from_csv(csv_path, **kwargs):
    """Find and load a path from a CSV file.

    Returns
    -------
    path: pathlib.Path
        The Path object matching the filters specifed as parameters.

    Examples
    --------
    Say you have a CSV file with the following format:

        subject,condition,data,path
        ...
        Doppio,REC-0+2,lf.bin,/Volumes/neuropixel_archive/Data/chronic/CNPIX4-Doppio/raw/3-18-2020_g0/3-18-2020_g0_imec0/3-18-2020_g0_t3.imec0.lf.bin
        ...

    Load like:
        get_datapath_from_csv(subject="Doppio", condition="REC-0+2", data="lf.bin")
    """
    df = pd.read_csv(csv_path)
    mask = pd.Series(True, index=df.index)
    for column, value in kwargs.items():
        mask = mask & (df[column] == value)

    return Path(df[mask]["path"].iloc[0])


def parse_sglx_stem(stem):
    """Parse recording identifiers from a SpikeGLX style filename stem.

    Paramters
    ---------
    stem: str
        The filename stem to parse, e.g. "my-run-name_g0_t1.imec2"

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
    """
    x = re.search(r"_g\d+_t\d+\.imec\d+\Z", stem)  # \Z forces match at string end.
    run = stem[: x.span()[0]]  # The run name is everything before the match
    gate = re.search(r"g\d+", x.group()).group()
    trigger = re.search(r"t\d+", x.group()).group()
    probe = re.search(r"imec\d+", x.group()).group()
    return (run, gate, trigger, probe)


def get_sglx_style_filename(run, gate, trigger, probe, ext):
    """Get SpikeGLX-style filename from parts.
    Note that ext is of the form `lf.bin`, not `.lf.bin`.
    """
    return f"{run}_{gate}_{trigger}.{probe}.{ext}"


def get_sglx_style_parent_path(run, gate, trigger, probe, root_dir):
    """Get the parent path where an SpikeGLX file would be found, assuming
    folder-per-probe organization.
    """
    probe_dir = f"{run}_{gate}_{probe}"
    run_dir = f"{run}_{gate}"
    return root_dir / run_dir / probe_dir


def get_sglx_style_abs_path(stem, ext, root):
    """Get the absolute path where a SpikeGLX filew ould be found, assuming
    folder-per-probe organization.

    root: pathlib.Path
        The absolute path to the directory containing the SGLX run directories.
    """
    run, gate, trigger, probe = parse_sglx_stem(stem)
    fname = get_sglx_style_filename(run, gate, trigger, probe, ext)
    parent = get_sglx_style_parent_path(run, gate, trigger, probe, root)
    return parent / fname


def get_sglx_style_datapaths(yaml_path, subject, experiment, condition, ext):
    """Get all datapaths, assuming a properly formatted YAML file an folder-per-probe
    organization."""
    with open(yaml_path) as fp:
        yaml_data = yaml.safe_load(fp)

    if (ext == "lf.bin") or (ext == "ap.bin"):
        root = Path(yaml_data[subject][experiment]["raw-data-root"])
    else:
        root = Path(yaml_data[subject][experiment]["analysis-root"])

    condition_manifest = list(flatten(yaml_data[subject][experiment][condition]))

    return [get_sglx_style_abs_path(stem, ext, root) for stem in condition_manifest]


def get_datapath(yaml_path, file, subject, experiment, condition=None):
    with open(yaml_path) as fp:
        yaml_data = yaml.safe_load(fp)

    datapath = Path(yaml_data[subject][experiment]["analysis-root"])

    if condition:
        datapath = datapath / condition

    return datapath / file
