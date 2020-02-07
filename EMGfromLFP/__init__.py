import os.path
from pathlib import Path

import numpy as np

from sleepscore import load

from .EMG import compute_EMG
from . import resample


EMGCONFIGKEYS = [
    'LFP_binPath', 'LFP_datatype', 'overwrite', 'EMGdata_savePath', 'sf',
    'window_size', 'bandpass', 'bandstop', 'LFP_downsample', 'LFP_chanList',
    'LFP_chanListType'
]


def load_EMG(EMGdatapath, tStart=None, tEnd=None, desired_length=None):
    """Load, slice and resample the EMG saved t `path`"""

    EMGmetapath = get_EMGmetapath(EMGdatapath)
    if not os.path.exists(EMGdatapath) or not os.path.exists(EMGmetapath):
        raise Exception(
            f"Couldn't find EMG files at {EMGdatapath}, {EMGmetapath}"
        )
    print(f"Load EMG at {EMGdatapath}")
    EMG_data = np.load(EMGdatapath)
    EMG_metadata = load.utils.load_yaml(EMGmetapath)

    # Select time segment of interest for EMG
    sf = EMG_metadata['sf']
    if tStart is None:
        tStart = 0.0
    firstSamp = int(tStart * sf)
    if tEnd is None:
        lastSamp = EMG_data.shape[1]
    else:
        lastSamp = int(tEnd * sf)
    print(f"Select timepoints between samples {firstSamp} and {lastSamp} out of"
          f" {EMG_data.shape[1]}")
    EMG_data = EMG_data[:, firstSamp:lastSamp+1]

    if desired_length is not None:
        print(f"Resampling EMG slice from {EMG_data.shape[1]} to "
              f"{desired_length} datapoints")
        EMG_data = resample.signal_resample(
            EMG_data[0, :],
            desired_length=desired_length,
            method='numpy'
        ).reshape((1, desired_length))

    return EMG_data, EMG_metadata


def run(EMG_config):
    """Compute and save the lfp-derived EMG and its metadata.

    Args:
        EMG_config: config dictionary containing the following keys:
            TODO
    """

    # Validate params
    assert set(EMG_config.keys()) == set(EMGCONFIGKEYS), (
        f'Expected EMG_config entries: {EMGCONFIGKEYS}'
    )

    # Get paths
    binPath = Path(EMG_config['LFP_binPath'])
    assert os.path.exists(binPath), "Data not found at {binPath}"
    if not EMG_config['EMGdata_savePath']:
        EMGdatapath = Path(binPath.parent / (binPath.stem + ".derivedEMG.npy"))
    else:
        EMGdatapath = Path(EMG_config['EMGdata_savePath'])
    EMGmetapath = get_EMGmetapath(EMGdatapath)

    # Do we compute the EMG?
    if os.path.exists(EMGdatapath) and os.path.exists(EMGmetapath):
        print(f"Found preexisting EMG files at {EMGdatapath}, {EMGmetapath}!")
        if EMG_config['overwrite']:
            print('--> Recomputing EMG ( `overwrite` == True )')
        else:
            print('--> Exiting ( `overwite` == False )')
            return

    # Compute EMG
    print(f"Computing EMG from LFP:")
    print("Loading LFP for EMG computing")
    # Load LFP for channels of interest
    lfp, sf, chanLabels, _ = load.loader_switch(
        binPath,
        datatype=EMG_config['LFP_datatype'],
        chanList=EMG_config['LFP_chanList'],
        chanListType=EMG_config['LFP_chanListType'],
        downSample=EMG_config['LFP_downsample'],
        tStart=None,
        tEnd=None,
    )
    print(f"Using the following channels for EMG derivation (labels): "
          f"{' - '.join(chanLabels)}")
    print("Computing EMG from LFP")
    EMG_data = compute_EMG(
        lfp, sf,
        EMG_config['sf'], EMG_config['window_size'], EMG_config['bandpass'],
        EMG_config['bandstop']
    )

    # Generate EMG metadata
    EMG_metadata = EMG_config.copy()
    EMG_metadata['EMGdatapath'] = str(EMGdatapath)
    EMG_metadata['EMGmetapath'] = str(EMGmetapath)
    EMG_metadata['LFP_chanLabels'] = chanLabels
    # EMG_metadata['gitcommit'] = subprocess.check_output(
    #     ["git", "describe"]
    # ).strip()

    # Save EMG
    print(f"Saving EMG metadata at {EMGmetapath}")
    load.utils.save_yaml(EMGmetapath, EMG_metadata)
    print(f"Saving EMG data at {EMGdatapath}")
    np.save(EMGdatapath, EMG_data)

    return EMG_data, EMG_metadata


def get_EMGmetapath(EMGdatapath):
    EMGdatapath = Path(EMGdatapath)
    metaName = EMGdatapath.stem + ".meta.yml"
    return Path(EMGdatapath.parent / metaName)
