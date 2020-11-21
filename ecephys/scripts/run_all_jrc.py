import yaml
import time

import os.path
from pathlib import Path

from ecephys.helpers import path_utils
from ecephys.sorters.jrc import jrc

# """""""""""""""""""""
# USER INPUT
# """""""""""""""""""""

cfg = {}

cfg['NPX_DIRNAME'] = 'processed_catGT_df'
cfg['NPX_DIRECTORIES'] = [
    # f'/Volumes/scratch/neuropixels/data/CNPIX2-Segundo/BL_15hs/{cfg["NPX_DIRNAME"]}',
    # f'/Volumes/scratch/neuropixels/data/CNPIX2-Segundo/BL_21hs/{cfg["NPX_DIRNAME"]}',
    # f'/Volumes/scratch/neuropixels/data/CNPIX2-Segundo/SR_15hs/{cfg["NPX_DIRNAME"]}',
    # f'/Volumes/scratch/neuropixels/data/CNPIX2-Segundo/SR_21hs/{cfg["NPX_DIRNAME"]}',

    # f'/Volumes/scratch/neuropixels/data/CNPIX3-Valentino/BL_15hs/{cfg["NPX_DIRNAME"]}',
    # f'/Volumes/scratch/neuropixels/data/CNPIX3-Valentino/BL_21hs/{cfg["NPX_DIRNAME"]}',
    # f'/Volumes/scratch/neuropixels/data/CNPIX3-Valentino/SR_15hs/{cfg["NPX_DIRNAME"]}',
    # f'/Volumes/scratch/neuropixels/data/CNPIX3-Valentino/SR_21hs/{cfg["NPX_DIRNAME"]}',

    # f'/Volumes/scratch/neuropixels/data/CNPIX4-Doppio/BL_15hs/{cfg["NPX_DIRNAME"]}',
    # f'/Volumes/scratch/neuropixels/data/CNPIX4-Doppio/BL_21hs/{cfg["NPX_DIRNAME"]}',
    # f'/Volumes/scratch/neuropixels/data/CNPIX4-Doppio/BL2_15hs/{cfg["NPX_DIRNAME"]}',
    # f'/Volumes/scratch/neuropixels/data/CNPIX4-Doppio/SR_15hs/{cfg["NPX_DIRNAME"]}',
    # f'/Volumes/scratch/neuropixels/data/CNPIX4-Doppio/SR_21hs/{cfg["NPX_DIRNAME"]}',

    # f'/Volumes/scratch/neuropixels/data/CNPIX5-Alessandro/BL_15hs/{cfg["NPX_DIRNAME"]}',
    # f'/Volumes/scratch/neuropixels/data/CNPIX5-Alessandro/BL_21hs/{cfg["NPX_DIRNAME"]}',
    # f'/Volumes/scratch/neuropixels/data/CNPIX5-Alessandro/BL2_15hs/{cfg["NPX_DIRNAME"]}',
    # f'/Volumes/scratch/neuropixels/data/CNPIX5-Alessandro/SR_15hs/{cfg["NPX_DIRNAME"]}',
    # f'/Volumes/scratch/neuropixels/data/CNPIX5-Alessandro/SR_21hs/{cfg["NPX_DIRNAME"]}',
]
cfg['RUN_DIR_PREFIX'] = 'catgt_'  # Run dirs in base directories start with this

cfg['JRC_PARAMS'] = {
    'CARMode': 'none',  # No need to CAR if we already ran catGT
    'refracInt': '1.0',
}

# Relative path to the yaml file containing 0-indexed list of bad channels used
# in the catGT run.
# - The path is relative to the directory containing the raw .bin data
# - If None: don't use bad channels
cfg['bad_channel_path'] = './bad_channels_thres=18/bad_channels.imec{prb_i}.ap.yml'

rerun_existing = False

n_jobs = 5

# """""""""""""""""""""
# END user input
# """""""""""""""""""""


def run_npx_dir(npx_directory, cfg):

    print('==========================================')
    print(f'Sort directory {npx_directory}')
    print('==========================================')

    run_specs = path_utils.get_run_specs(
        npx_directory, run_dir_prefix=cfg['RUN_DIR_PREFIX']
    )
    print(f'Run specs for npx dir. Looks good?: {run_specs}')

    # Runs
    for run_name, gate_str, triggers_i, probes_i in run_specs:

        # Probes
        for prb_i in probes_i:

            # build path to the raw data folder
            run_folder_name = cfg['RUN_DIR_PREFIX'] + run_name + '_g' + gate_str
            prb_fld_name = run_name + '_g' + gate_str + '_imec' + str(prb_i)
            prb_fld = Path(npx_directory)/run_folder_name/prb_fld_name
            print(f"Processing run/probe: {prb_fld_name}")

            # Load bad channels for that probe/stream
            if cfg['bad_channel_path'] is None:
                bad_channels = None
            else:
                # Go to raw data dir
                raw_run_name = run_name + '_g' + gate_str
                raw_prb_fld_name = raw_run_name + '_imec' + str(prb_i)
                raw_prb_fld = Path(npx_directory).parent/'raw'/raw_run_name/raw_prb_fld_name
                badchan_path = raw_prb_fld/cfg['bad_channel_path'].format(**{
                    'prb_i': prb_i,
                })
                if not badchan_path.exists():
                    raise ValueError(
                        "Could not find yaml file containing bad channels at:"
                        f"\n{badchan_path}"
                    )
                with open(badchan_path, 'r') as f:
                    bad_channels = yaml.load(f, Loader=yaml.FullLoader)
                print(f"Specifying N={len(bad_channels)} bad channels")
            # JRC bad channels are 1-indexed
            badChans = [int(c) + 1 for c in bad_channels]

            # Single trigger in the probe directory (`tcat`)
            if len(triggers_i) > 1:
                raise NotImplementedError
            trigger_i = triggers_i[0]

            # Bin file
            binfilename = \
                f'{run_name}_g{gate_str}_t{trigger_i}.imec{prb_i}.ap.bin'
            binpath = prb_fld/binfilename
            assert os.path.exists(binpath)

            # JRC output
            output_dirname = 'jrc_' + '_'.join(
                sorted([
                    f'{key}={value}' for key, value in cfg['JRC_PARAMS'].items()
                ])
            )
            jrc_output_dir = prb_fld/output_dirname
            jrc_output_dir.mkdir(exist_ok=True)
            config_name = f'{run_name}_g{gate_str}_t{trigger_i}.imec{prb_i}'

            # Already run?
            if (
                    (jrc_output_dir/f'{config_name}_res.mat').exists()
                    and not rerun_existing
            ):
                print('found JRC result file in {jrc_output_dir}: passing')
                continue

            # Run JRC

            tstart = time.time()
            jrc.run_JRC(
                binpath,
                jrc_output_dir,
                jrc_params=cfg['JRC_PARAMS'],
                badChans=badChans,
                config_name=config_name,
            )
            print(f'Done sorting porbe {prb_fld_name}: ', end='')
            print(f'Total time={(time.time() - tstart)/60}min')

    print(f'Done sorting directory {npx_directory}')
    print('==========================================\n')


def main():

    if n_jobs == 1:
        for npx_dir in cfg['NPX_DIRECTORIES']:
            run_npx_dir(npx_dir, cfg)
    else:
        from joblib import Parallel, delayed

        Parallel(
            n_jobs=n_jobs
        )(
            delayed(run_npx_dir)(npx_dir, cfg)
            for npx_dir in cfg['NPX_DIRECTORIES']
        )


if __name__ == '__main__':
    main()
