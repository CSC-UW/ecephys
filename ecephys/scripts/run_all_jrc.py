import os.path
from pathlib import Path

from ecephys.utils import path_utils
from ecephys.sorters.jrc import jrc

# """""""""""""""""""""
# USER INPUT
# """""""""""""""""""""

BASE_DIRECTORIES = [
    "/Volumes/scratch/neuropixels/data/CNPIX4/3-4PM/BL/processed",
    "/Volumes/scratch/neuropixels/data/CNPIX4/3-4PM/SR/processed",
    "/Volumes/scratch/neuropixels/data/CNPIX4/9-10PM/BL/processed",
    "/Volumes/scratch/neuropixels/data/CNPIX4/9-10PM/SR/processed",
    "/Volumes/scratch/neuropixels/data/CNPIX3/3-4PM/BL/processed",
    "/Volumes/scratch/neuropixels/data/CNPIX3/3-4PM/BL2/processed",
    "/Volumes/scratch/neuropixels/data/CNPIX3/3-4PM/SR/processed",
    "/Volumes/scratch/neuropixels/data/CNPIX3/9-10PM/BL/processed",
    "/Volumes/scratch/neuropixels/data/CNPIX3/9-10PM/SR/processed",
]  # Each contains the run directories
RUN_DIRECTORY_PREFIX = "catgt_"  # Run dirs in base directories start with this

JRC_PARAMS = {
    "CARMode": "none",  # No need to CAR if we already ran catGT
    "refracInt": "1.0",
}

badChannels = [
    [192],
    [192],
    [192],
    [192],
    [192],
    [192],
    [192],
    [192],
    [192],
]  # Same length as BASE_DIRECTORIES, 1-indexed !!!

rerun_existing = False

n_jobs = 3

# """""""""""""""""""""
# END user input
# """""""""""""""""""""

assert len(badChannels) == len(BASE_DIRECTORIES)


def run_base_dir(i, base_dir):

    print(f"\n\nRunning dir {base_dir}")

    run_specs = path_utils.get_run_specs(base_dir, run_dir_prefix=RUN_DIRECTORY_PREFIX)
    print(f"Run specs for base dir: {run_specs}")

    # Bad chans
    badChans = badChannels[i]

    # Runs
    for run_name, gate_i, triggers_i, probes_i in run_specs:

        run_dir = Path(base_dir) / f"{RUN_DIRECTORY_PREFIX}{run_name}_g{gate_i}"

        # Probes
        for probe_i in probes_i:

            probe_dir = run_dir / f"{run_name}_g{gate_i}_imec{probe_i}"

            # Single trigger in the probe directory (`tcat`)
            if len(triggers_i) > 1:
                raise NotImplementedError
            trigger_i = triggers_i[0]

            # Bin file
            binfilename = f"{run_name}_g{gate_i}_t{trigger_i}.imec{probe_i}.ap.bin"
            binpath = probe_dir / binfilename
            assert os.path.exists(binpath)

            # JRC output
            output_dirname = "jrc_" + "_".join(
                sorted([f"{key}={value}" for key, value in JRC_PARAMS.items()])
            )
            jrc_output_dir = probe_dir / output_dirname
            jrc_output_dir.mkdir(exist_ok=True)
            config_name = f"{run_name}_g{gate_i}_t{trigger_i}.imec{probe_i}"

            # Already run?
            if (
                jrc_output_dir / f"{config_name}_res.mat"
            ).exists() and not rerun_existing:
                print("found JRC result file in {jrc_output_dir}: passing")
                continue

            # Run JRC
            jrc.run_JRC(
                binpath,
                jrc_output_dir,
                jrc_params=JRC_PARAMS,
                badChans=badChans,
                config_name=config_name,
            )


def main():

    if n_jobs == 1:
        for i, base_dir in enumerate(BASE_DIRECTORIES):
            run_base_dir(i, base_dir)
    else:
        from joblib import Parallel, delayed

        Parallel(n_jobs=n_jobs)(
            delayed(run_base_dir)(i, base_dir)
            for i, base_dir in enumerate(BASE_DIRECTORIES)
        )


if __name__ == "__main__":
    main()
