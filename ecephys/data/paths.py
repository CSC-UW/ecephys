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


def get_sglx_style_filename(run, gate, trigger, probe, ext, catgt_data=False):
    """Get SpikeGLX-style filename from parts.
    Note that ext is of the form `lf.bin`, not `.lf.bin`.
    """
    trig = trigger if not catgt_data else "tcat"
    return f"{run}_{gate}_{trig}.{probe}.{ext}"


def get_sglx_style_parent_path(run, gate, trigger, probe, root_dir, catgt_data=False):
    """Get the parent path where an SpikeGLX file would be found, assuming
    folder-per-probe organization.
    """
    probe_dir = f"{run}_{gate}_{probe}"
    run_dir = f"{run}_{gate}" if not catgt_data else f"catgt_{run}_{gate}"
    return root_dir / run_dir / probe_dir


def get_sglx_style_abs_path(stem, ext, root, catgt_data=False):
    """Get the absolute path where a SpikeGLX filew ould be found, assuming
    folder-per-probe organization.

    CatGT saves data with `catgt_` prepended to the run directory and `cat`
    as trigger idx
    """
    run, gate, trigger, probe = parse_sglx_stem(stem)
    fname = get_sglx_style_filename(
        run, gate, trigger, probe, ext, catgt_data=catgt_data
    )
    parent = get_sglx_style_parent_path(
        run, gate, trigger, probe, root, catgt_data=catgt_data
    )
    return parent / fname


def get_sglx_style_datapaths(yaml_path, subject, condition, ext, catgt_data=False, cat_trigger=False,
                             data_root=None):
    """Get all datapaths, assuming a properly formatted YAML file an folder-per-probe
    organization.

    The data for each condition is loaded in one of three ways:

    
    Kwargs:
        catgt_data: bool
            Path to corresponding catGT-processed concatenated file.
            catGT file is saved in the analysis directory, uses `cat`
            as the trigger index, and has `catgt_` prepended to the
            run directory.
        cat_trigger: bool
            Replace trigger id with 'cat' to designate data issued from concatenated files.
        data-root: str
            Force root path
    """
    with open(yaml_path) as fp:
        yaml_data = yaml.safe_load(fp)

    data_file = any([
        ext in data_ext
        for data_ext in ["lf.bin", "lf.meta", "ap.bin", "ap.meta"]
    ]) 
    if data_root is not None:
        root = Path(data_root)
    elif data_file and not catgt_data:
        root = Path(yaml_data[subject]["raw-data-root"])
    else:
        root = Path(yaml_data[subject]["analysis-root"])

    condition_data = yaml_data[subject][condition]

    # How do we interpret the condition's data?
    if isinstance(condition_data, dict):
        combined_condition = False
        # subject: condition: experiment_id: [stem_0, stem_1]
        # Append experiment id to root
        paths = []
        for experiment_id in condition_data.keys():
            condition_manifest = list(flatten(condition_data[experiment_id]))
            # All elements should be raw data stems
            assert all(['.imec' in stem for stem in condition_manifest])
            experiment_root = root / experiment_id
            paths += [
                get_sglx_style_abs_path(
                    stem, ext, experiment_root, catgt_data=catgt_data
                )
                for stem in condition_manifest
            ]
    else:
        condition_manifest = list(flatten(condition_data))
        if all(['.imec' in stem for stem in condition_manifest]):
            # subject: condition:[stem_0, stem_1]
            # All elements are raw data stems
            combined_condition = False
            paths = [
                get_sglx_style_abs_path(stem, ext, root, catgt_data=catgt_data)
                for stem in condition_manifest
            ]
        elif all([cond in yaml_data[subject] for cond in condition_manifest]):
            combined_condition = True
            # subject: combined_condition: [cond1, cond2]
            # Recursive loading
            paths = []
            for cond in condition_manifest:
                paths += get_sglx_style_datapaths(
                    yaml_path, subject, cond, ext, catgt_data=catgt_data,
                    cat_trigger=cat_trigger, data_root=data_root
                )
        else:
            raise ValueError(f"Incorrect format for {subject}, {condition}")

    # For catGT data the triggers should have been concatenated
    if catgt_data and not combined_condition:
        assert len(set(paths)) == 1
        paths = list(set(paths))

    # No duplicates
    if len(paths) != len(set(paths)):
        raise ValueError(f"Duplicates in requested paths: {paths}")

    return paths


def get_datapath(yaml_path, subject, condition, file):
    with open(yaml_path) as fp:
        yaml_data = yaml.safe_load(fp)

    datapath = Path(yaml_data[subject]["analysis-root"])

    if condition:
        datapath = datapath / condition

    return datapath / file
