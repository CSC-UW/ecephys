"""Manage paths and directories for neuropixels data.

All run directories are assumed to have been saved with probe folders.
"""

import pathlib


def get_run_specs(raw_data_dir, run_dir_prefix=None):
    """Return info about runs in raw directory.

    Example::
        run_specs = [
            ['3-17-2020', '0', '3,3', '0']
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
        run_dir_prefix = ''

    raw_data_dir = pathlib.Path(raw_data_dir)

    print(f'Generating run_specs for raw data directory: `{raw_data_dir}`')
    run_specs = []

    # Iterate on run directories
    for run_dir in [
        dir.name for dir in raw_data_dir.iterdir()
        if dir.is_dir()
        and dir.name.startswith(run_dir_prefix)
    ]:
        # Remove prefix
        run_dir_name = run_dir[len(run_dir_prefix):]

        # Check it's a run directory and parse out run_name/gate_name
        if not len(run_dir_name.split('_g')) == 2:
            print(f'Passing subdir `{run_dir}`: not a run directory')
            continue
        run_name, gate_i = run_dir_name.split('_g')
        assert gate_i.isdigit()

        probes = []
        probe_triggers = []

        # Iterate on probe directories
        for probe_dir in [
            dir.name for dir in (raw_data_dir/run_dir).iterdir() if dir.is_dir()
        ]:

            if not (
                len(probe_dir.split('_imec')) == 2
                and probe_dir.split('_imec')[0].split('_g')[0] == run_name
            ):
                print(f'Passing subdir `{probe_dir}`: not a probe directory')
                continue

            _, probe_i = probe_dir.split('_imec')
            assert probe_i.isdigit()

            probes.append(probe_i)

            # Iterate on triggers
            triggers = []
            for bin_stem in [
                    binfile.name.split('.')[0]
                    for binfile in (raw_data_dir/run_dir/probe_dir).iterdir()
                    if '.bin' in binfile.suffixes
                    and '.ap' in binfile.suffixes
            ]:

                assert len(bin_stem.split('_t')) == 2

                triggers.append(bin_stem.split('_t')[1])

            probe_triggers.append(triggers)

        # Sanity: check that we have the same set of triggers for all probes
        assert all([
            set(probe_trigs) == set(probe_triggers[0])
            for probe_trigs in probe_triggers
        ]), "Set of triggers differ across probe directories"

        triggers = probe_triggers[0]

        run_specs.append(
            [run_name, gate_i, sorted(triggers), sorted(probes)]
        )

    return run_specs
