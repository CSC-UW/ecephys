import os
import subprocess
import time
from pathlib import Path

import numpy as np

import yaml
from ecephys.utils.path_utils import get_allen_formatted_run_specs
from ecephys_spike_sorting.scripts.utils import SpikeGLX_utils

# script to run CatGT,  on data collected using # SpikeGLX. The construction of
# the paths assumes data was saved with "Folder per probe" selected (probes
# stored in separate folders) AND
# that CatGT is run with the -out_prb_fld option

# For each directory <processed_dir> in BASE_DIRECTORIES, this computes catGT
# for all the data located at <processed_dir>/../raw

# -------------------------------
# -------------------------------
# User input -- Edit this section
# -------------------------------
# -------------------------------

# Directory in which the catGT run is saved.
# - If you change catGT parameters, you may want to change the output directory
# name (eg `../processed` or `../processed_nobadchans`)
# - catGT output will be in the standard SpikeGLX directory structure:
# run_folder/probe_folder/*.bin
# - The raw data used by catGT is assumed to be located at <output_dir>/../raw
OUTPUT_DIRECTORIES = [
    "/Volumes/neuropixel_archive/Data/chronic/CNPIX2-Segundo/BL_15hs/processed_catGT_nobadchan_df_params",
    # '/Volumes/scratch/neuropixels/data/CNPIX4/3-4PM/SR/processed',
]

# Relative path to the yaml file containing 0-indexed list of bad channels used
# in the catGT run.
# - The path is relative to the directory containing the raw .bin data
# - If None: don't use bad channels
bad_channel_path = "./bad_channels/bad_channels.imec{prb_i}.{stream}.yml"
bad_channel_path = None

# CatGT params. Incorporated in command string:
# catGT_cmd_string = '-prb_fld -out_prb_fld -aphipass=300 -aplopass=9000 -gbldmx -gfix=0,0.10,0.02 -SY=0,384,6,500 -SY=1,384,6,500'
catGT_stream = "-ap"  # -lf not supported for now
catGT_hipass = 300
catGT_lopass = 9000
catGT_gfix = "0,0.10,0.02"

# 0-indexed. Should be same id for all runs and probes
sync_chan_id = 384

# Paths
wineexe_fullpath = "/usr/bin/wine"  # Run with wine if not None
catGTexe_fullpath = r"/Volumes/scratch/neuropixels/bin/CatGT/CatGT.exe"

rerun_existing = True
assert rerun_existing == True  # TODO

n_jobs = 2

# -------------------------------
# -------------------------------
# END USER INPUT
# -------------------------------
# -------------------------------

assert catGT_stream == "-ap"


def call_catGT_cmd(
    npx_directory,
    output_dir,
    run_name,
    gate_str,
    prb_i,
    trigger_str,
    bad_chans=None,
    stream_str=catGT_stream,
    catGTexe_fullpath=catGTexe_fullpath,
    winexe_fullpath=None,
    catGT_hipass=catGT_hipass,
    catGT_lopass=catGT_lopass,
    catGT_gfix=catGT_gfix,
    sync_chan_id=sync_chan_id,
    logPath=None,
):

    # ------------
    # CatGT cmd string
    # ------------

    # CatGT command string includes all instructions for catGT operations
    # Note 1: directory naming in this script requires -prb_fld and -out_prb_fld
    # Note 2: this command line includes specification of edge extraction for each probe. The
    # sync channel idx is assumed to be 384 for each probe
    # Note 3: This command line ignores bad channels for each probe data
    # Note 3: This command line ignores bad channels
    # see CatGT readme for details

    catGT_cmd_str = (
        f"-prb_fld -out_prb_fld "
        f"-aphipass={catGT_hipass} -aplopass={catGT_lopass} "
        f"-gbldmx "
        f"-gfix={catGT_gfix} "
        f"-SY={prb_i},{sync_chan_id},6,500"
    )

    cmd_parts = list()

    if wineexe_fullpath:
        # Run the command with wine if we're on linux
        cmd_parts.append(wineexe_fullpath)
    cmd_parts.append(catGTexe_fullpath)
    cmd_parts.append("-dir=" + str(npx_directory))
    cmd_parts.append("-dest=" + str(output_dir))
    cmd_parts.append("-run=" + run_name)
    cmd_parts.append("-g=" + gate_str)
    cmd_parts.append("-t=" + trigger_str)
    cmd_parts.append("-prb=" + str(prb_i))
    cmd_parts.append(stream_str)
    cmd_parts.append(catGT_cmd_str)
    if bad_chans is not None:
        bad_chans_str = ",".join(bad_chans)
        cmd_parts.append("-chnexcl=" + bad_chans_str)

    catGT_cmd = " "  # use space as the separator for the command parts
    catGT_cmd = catGT_cmd.join(cmd_parts)

    print("CatGT command line:" + catGT_cmd)

    Path(logPath.parent).mkdir(parents=True, exist_ok=True)
    with open(logPath, "a") as f:
        f.write(catGT_cmd + "\n")

    start = time.time()
    subprocess.call(catGT_cmd, shell=True)

    execution_time = time.time() - start

    print("total time: " + str(np.around(execution_time, 2)) + " seconds")


def run_catGT(output_dir):

    print("==========================================")
    print(f"Run catGT in output directory {output_dir}")
    print("==========================================")

    output_dir = Path(output_dir)

    # Raw data directory = npx_directory
    # run_specs = name, gate, trigger and probes to process
    npx_directory = str(output_dir.parent / "raw")
    if not Path(npx_directory).exists():
        raise ValueError(
            f"Could not find raw data directory at `<output_dir>/../raw` for"
            f" output_dir {output_dir}\n"
            f"`<output_dir>/../raw = {npx_directory}"
        )
    print(f"Raw npix data directory: {npx_directory}")

    # Each run_spec is a list of 4 strings:
    #   undecorated run name (no g/t specifier, the run field in CatGT)
    #   gate index, as a string (e.g. '0')
    #   triggers to process/concatenate, as a string e.g. '0,400', '0,0 for a single file
    #           can replace first limit with 'start', last with 'end'; 'start,end'
    #           will concatenate all trials in the probe folder
    #   probes to process, as a string, e.g. '0', '0,3', '0:3'
    # run_specs = [
    #     ['my_run', '0', '0,11', '0'],
    #     ...
    # ]
    run_specs = get_allen_formatted_run_specs(npx_directory)
    print(f"run specs in npx directory. Looks good?: {run_specs}\n\n\n")
    if not len(run_specs):
        raise ValueError(f"Could not find runs in npx directory `npx_directory`")

    # Get streams
    if catGT_stream == "-ap":
        stream = "ap"
    else:
        raise ValueError("Only -ap stream supported for now")

    catGT_logpath = output_dir / "logs" / "CatGT.log"

    if rerun_existing:
        try:
            os.remove(catGT_logpath)
        except OSError:
            pass

    output_dir.mkdir(exist_ok=True, parents=True)

    for spec in run_specs:

        run_name = spec[0]
        gate_str = spec[1]

        # Get probes ids for that run
        probes_i = spec[3].split(",")

        # Run catGT separately for each probe because the set of bad channels
        # may change
        for prb_i in probes_i:

            # build path to the raw data folder
            run_folder_name = spec[0] + "_g" + gate_str
            prb_fld_name = run_folder_name + "_imec" + str(prb_i)
            prb_fld = Path(npx_directory) / run_folder_name / prb_fld_name
            print(f"Processing run/probe: {prb_fld_name}")

            # Load bad channels for that probe/stream
            if bad_channel_path is None:
                bad_channels = None
            else:
                badchan_path = prb_fld / bad_channel_path.format(
                    **{
                        "prb_i": prb_i,
                        "stream": stream,
                    }
                )
                if not badchan_path.exists():
                    raise ValueError(
                        "Could not find yaml file containing bad channels at:"
                        f"\n{badchan_path}"
                    )
                with open(badchan_path, "r") as f:
                    bad_channels = yaml.load(f)
                print("Specifying N={len(bad_channels)} bad channels")

            first_trig, last_trig = SpikeGLX_utils.ParseTrigStr(spec[2], prb_fld)
            trigger_str = repr(first_trig) + "," + repr(last_trig)

            # Run catGT
            call_catGT_cmd(
                npx_directory,
                output_dir,
                run_name,
                gate_str,
                prb_i,
                trigger_str,
                bad_chans=bad_channels,
                stream_str=catGT_stream,
                catGTexe_fullpath=catGTexe_fullpath,
                winexe_fullpath=wineexe_fullpath,
                catGT_hipass=catGT_hipass,
                catGT_lopass=catGT_lopass,
                catGT_gfix=catGT_gfix,
                sync_chan_id=sync_chan_id,
                logPath=catGT_logpath,
            )


def main():

    if n_jobs == 1:
        for output_dir in OUTPUT_DIRECTORIES:
            run_catGT(output_dir)
    else:
        from joblib import Parallel, delayed

        Parallel(n_jobs=n_jobs)(
            delayed(run_catGT)(output_dir) for output_dir in OUTPUT_DIRECTORIES
        )


if __name__ == "__main__":
    main()
