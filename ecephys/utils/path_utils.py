"""Manage paths and directories for neuropixels data.

All run directories are assumed to have been saved with probe folders.
"""
from typing import Union
import pathlib
import regex as re  # TODO: Shouldn't this just be import re?

Pathlike = Union[pathlib.Path, str]


# TODO: Is this still used, or should be removed?
def get_run_specs(raw_data_dir, run_dir_prefix=None):
    """Return info about runs in raw directory.

    Example::
        run_specs = [
            ['3-17-2020', '0', ['3','4','5'], ['0']],
            ['3-18-2020.my-run', '0', ['0'], ['0','1']],
        ]

    Args:
        raw_data_dir (str or path-like): path to the directory containing run
            directories. Data is assumed to be saved with probe directories.
            Triggers are derived from the presence of .ap bin files in the
            probe directories.

    Kwargs:
        run_dir_prefix (str or None): Prefix to the run directories name.
            Subdirectories in `raw_data_dir` are parsed as
            '{run_dir_prefix}{run_name}'. eg "catgt_"

    Returns:
        (list): List of lists each of the form::
                ['run_name', 'gate_i', 'triggers_i', 'probes_i']
            containing the following information parsed from files/directories:
              - undecorated run name (no g/t specifier, the run field in CatGT)
              - gate index, as a string (e.g. '0')
              - triggers as a list of strings e.g. ['0', '1'] or ['cat']
              - probes as a list of strings, e.g. ['0', '1']
    """
    if run_dir_prefix is None:
        run_dir_prefix = ""

    raw_data_dir = pathlib.Path(raw_data_dir)

    print(f"Generating run_specs for raw data directory: `{raw_data_dir}`")
    run_specs = []

    # Iterate on run directories
    for run_dir in [
        dir.name
        for dir in raw_data_dir.iterdir()
        if dir.is_dir() and dir.name.startswith(run_dir_prefix)
    ]:
        # Remove prefix
        run_dir_name = run_dir[len(run_dir_prefix) :]

        # Check it's a run directory and parse out run_name/gate_name
        run_dir_match = re.match("(.*)_g([0-9]+)", run_dir_name)
        if run_dir_match is None:
            print(f"Passing subdir `{run_dir}`: not a run directory")
            continue
        run_name, gate_i = run_dir_match.groups()

        probes = []
        probe_triggers = []

        # Iterate on probe directories
        for probe_dir in [
            dir.name for dir in (raw_data_dir / run_dir).iterdir() if dir.is_dir()
        ]:
            prb_dir_match = re.match(f"{run_dir_name}_imec([0-9]+)", probe_dir)
            if prb_dir_match is None:
                print(f"Passing subdir `{probe_dir}`: not a probe directory")
                continue
            probe_i = prb_dir_match.groups()[0]

            probes.append(probe_i)

            # Iterate on files and parse triggers
            triggers = []
            for binfile in [
                f.name
                for f in (raw_data_dir / run_dir / probe_dir).iterdir()
                if f.is_file()
            ]:
                bin_match = re.match(
                    f"\A{run_name}_g{gate_i}_t(.*).imec{probe_i}.ap.bin\Z",
                    binfile,
                )
                if bin_match is None:
                    # print(binfile)
                    # print( f'{run_name}_g{gate_i}_t([0-9]+).imec{probe_i}.ap.bin')
                    continue
                print(binfile)
                # print( f'{run_name}_g{gate_i}_t([0-9]+).imec{probe_i}.ap.bin')
                trg_i = bin_match.groups()[0]
                assert trg_i.isdigit() or trg_i == "cat"
                triggers.append(trg_i)

            # Sanity: signle matched file per trigger
            assert len(set(triggers)) == len(triggers)

            probe_triggers.append(triggers)

        # Sanity: check that we have the same set of triggers for all probes
        assert all(
            [
                set(probe_trigs) == set(probe_triggers[0])
                for probe_trigs in probe_triggers
            ]
        ), "Set of triggers differ across probe directories"

        triggers = probe_triggers[0]

        run_specs.append([run_name, gate_i, sorted(triggers), sorted(probes)])

    return run_specs


# TODO: Is this still used, or should be removed?
def get_allen_formatted_run_specs(raw_data_dir, run_dir_prefix=None):
    """Return info about runs formatted as expected in `ecephys_spike_sorting`.

    Example::
        run_specs = [
            ['3-17-2020', '0', '3,3', '0'],
            ['3-18-2020', '0', '0,11', '0,1,2'],
        ]
    """
    run_specs = get_run_specs(
        raw_data_dir, run_dir_prefix
    )  # All probe and trigger ids as list
    return [
        [
            spec[0],
            spec[1],
            f"{min(map(int, spec[2]))},{max(map(int, spec[2]))}",  # min(trg_i),max(trg_i), eg '0,11'
            ",".join(spec[3]),  # eg 0,1,2
        ]
        for spec in run_specs
    ]  # Format expected in ecephys_spike_sorting scripts
