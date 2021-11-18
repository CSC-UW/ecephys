import time
from pathlib import Path

import numpy as np

import yaml
from ecephys.utils.path_utils import get_run_specs
from ecephys.sglx import get_xy_coords
from ecephys.signal.bad_channels import get_good_channels
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

# Directory in which raw data is saved
# - Assumes standard SpikeGLX directory structure: run_folder/probe_folder/*.bin
NPX_DIRECTORIES = [
    # "/Volumes/neuropixel_archive/Data/chronic/CNPIX2-Segundo/raw",
]

# Channel marked as bad and excluded from analysis for each
# TODO: What if changes across runs?
ignored_channels = [191]

# Threshold for RMS relative to neighbors median. 15..20 . Lower is less conservative for exclusion
noise_threshold = 18

# Relative path to the created yaml files containing 0-indexed list of bad
# channels for each trigger
# - The path is relative to the directory containing the raw .bin data
# - We also create a file containing the union (`..tunion..``) or intersection
# (`..tinter`..) of bad channels created for each trigger
# BAD_CHANNEL_PATH = "./bad_channels/bad_channels_t{trigger_i}.imec{prb_i}.{stream}.yml"
BAD_CHANNEL_PATH = (
    "./bad_channels_thres="
    + str(noise_threshold)
    + "/bad_channels_t{trigger_i}.imec{prb_i}.{stream}.yml"
)
NCHAN_FILENAME = "N_bad_channels.imec{prb_i}.{stream}.yml"  # Number of bad channels for each trigger. Saved in same directory as bad channel files
TRIG_FILENAME = "trg_per_chan.imec{prb_i}.{stream}.yml"  # List of triggers for which a channel is bad

# Streams for which we compute bad channels
streams = ["ap"]

n_jobs = 1
# overwrite = True
overwrite = False


def run_bad_channels_detection(npx_directory):
    """Create bad channels files for each trigger and for whole run."""

    print("==========================================")
    print("==========================================")
    print(f"Detect bad channels for npx directory {npx_directory}")
    print("==========================================")
    print("==========================================")

    npx_directory = Path(npx_directory)
    if not npx_directory.exists():
        raise ValueError(f"Could not find raw data directory at {npx_directory}")
    print(f"Raw npix data directory: {npx_directory}")

    # Each run_spec is a list containing:
    #   - undecorated run name (no g/t specifier, the run field in CatGT)
    #   - gate index, as a string (e.g. "0")
    #   - triggers to process, as a list e.g. ["0", "1", "400"]
    #   - probes to process, as a list, e.g. ["0", "1"]
    # run_specs = [
    #     ["my_run", "0", ["0", "1", "400"], ["0", "1"]],
    #     ...
    # ]
    run_specs = get_run_specs(npx_directory)
    print(f"Detected runs. Looks good?: {run_specs}\n")

    for run_name, gate_str, triggers_i, probes_i in run_specs:

        for prb_i in probes_i:

            # build path to the raw data folder
            run_folder_name = run_name + "_g" + gate_str
            prb_fld_name = run_folder_name + "_imec" + str(prb_i)
            prb_fld = Path(npx_directory) / run_folder_name / prb_fld_name
            print(f"====================================")
            print(f"Processing run/probe: {prb_fld_name}")
            print(f"Saving output at : {Path(prb_fld_name)/BAD_CHANNEL_PATH}")
            print(f"====================================")

            tstart = time.time()

            for stream in streams:

                # Pass if files already there
                if not overwrite:
                    nchan_path = (prb_fld / BAD_CHANNEL_PATH.format(
                        **{"prb_i": prb_i, "stream": stream, "trigger_i": "NULL"}
                        )).parent/NCHAN_FILENAME.format(
                            **{"prb_i": prb_i, "stream": stream}
                        )
                    ).parent / NCHAN_FILENAME.format(
                        **{"prb_i": prb_i, "stream": stream}
                    )
                    if nchan_path.exists():
                        print(f"overwrite==False: Passing stream {stream}, run/probe {prb_fld_name}")
                        continue

                # Fill bad channels for each trigger to get whole-run bad
                # channels
                all_trg_bad_channels = []
                # Number of bad chans for report file
                all_trg_N_bad = {}  # {trg: N_badchans}
                # list of triggers for which a channel is bad
                all_bad_trg = {chan_i: [] for chan_i in range(384)}

                for trg_i in triggers_i:

                    raw_data_file = (
                        prb_fld
                        / f"{run_name}_g{gate_str}_t{trg_i}.imec{prb_i}.{stream}.bin"
                    )

                    (
                        probe_type,
                        sample_rate,
                        num_channels,
                        uVPerBit,
                    ) = SpikeGLX_utils.EphysParams(raw_data_file)

                    if stream == "ap":
                        (
                            ap_chans,  # AP channel indices
                            xcoords,
                            ycoords,
                        ) = get_xy_coords(raw_data_file)
                    else:
                        # Not implemented
                        assert 0

                    # Order channels along probe
                    channel_map = np.argsort(ycoords)

                    print(f"t={trg_i}: ", end="")
                    good_channels_mask = get_good_channels(
                        raw_data_file,
                        num_channels,
                        sample_rate,
                        uVPerBit,
                        noise_threshold=noise_threshold,
                        ignored_channels=ignored_channels,
                        channel_map=channel_map,
                        ap_lf_chans=ap_chans,
                    )

                    # There we go
                    bad_channels = sorted(np.where(~good_channels_mask)[0].tolist())

                    # Save
                    bad_path = prb_fld / BAD_CHANNEL_PATH.format(
                        **{"prb_i": prb_i, "stream": stream, "trigger_i": trg_i}
                    )
                    bad_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(bad_path, "w") as f:
                        yaml.dump(bad_channels, f, default_flow_style=False)

                    all_trg_bad_channels.append(bad_channels)
                    all_trg_N_bad[trg_i] = len(bad_channels)
                    for chan_i in bad_channels:
                        all_bad_trg[chan_i] += [trg_i]

                # Compute and save whole-run bad channels
                # Intersection of all triggers bad chans
                inter_bad_chans = sorted(
                    list(set.intersection(*map(set, all_trg_bad_channels)))
                )
                inter_path = prb_fld / BAD_CHANNEL_PATH.format(
                    **{"prb_i": prb_i, "stream": stream, "trigger_i": "inter"}
                )
                with open(inter_path, "w") as f:
                    yaml.dump(inter_bad_chans, f, default_flow_style=False)
                print(
                    f"Whole run bad N chans (intersection method): {len(inter_bad_chans)}"
                )
                all_trg_N_bad["inter"] = len(inter_bad_chans)
                # union of all triggers bad chans
                union_bad_chans = sorted(
                    list(set.union(*map(set, all_trg_bad_channels)))
                )
                union_path = prb_fld / BAD_CHANNEL_PATH.format(
                    **{"prb_i": prb_i, "stream": stream, "trigger_i": "union"}
                )
                with open(union_path, "w") as f:
                    yaml.dump(union_bad_chans, f, default_flow_style=False)
                print(
                    f"Whole run bad N chans (union method): {len(union_bad_chans)}"
                )
                all_trg_N_bad["union"] = len(union_bad_chans)

                # Save report file containing number of bad channels
                nchan_path = union_path.parent / NCHAN_FILENAME.format(
                    **{"prb_i": prb_i, "stream": stream}
                )
                with open(nchan_path, "w") as f:
                    yaml.dump(all_trg_N_bad, f, default_flow_style=False)
                print(f"Save Number of channels per trigger at {nchan_path}")
                # Save report file containing list of trigs for which each channel is bad
                badtrig_path = union_path.parent / TRIG_FILENAME.format(
                    **{"prb_i": prb_i, "stream": stream}
                )
                with open(badtrig_path, "w") as f:
                    yaml.dump(all_bad_trg, f, default_flow_style=False)
                print(
                    f"Save list of triggers for which each channel is bad at {badtrig_path}"
                )

            print(
                f"Done processing probe. Running time = {(time.time() - tstart)/60}min. \n"
            )


def main():

    if n_jobs == 1:
        for npx_dir in NPX_DIRECTORIES:
            run_bad_channels_detection(npx_dir)
    else:
        from joblib import Parallel, delayed

        Parallel(n_jobs=n_jobs)(
            delayed(run_bad_channels_detection)(npx_dir) for npx_dir in NPX_DIRECTORIES
        )


if __name__ == "__main__":
    main()
