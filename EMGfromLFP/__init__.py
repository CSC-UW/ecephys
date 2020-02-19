import os.path
from pathlib import Path

import numpy as np

from sleepscore import load

from .EMG import compute_EMG
from . import resample


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
    """Run `compute_and_save` from EMG config dictionary

    Args:
        EMG_config: config dictionary containing the following keys:
            TODO
    """
    print(f"EMG config = {EMG_config}, \n")

    # Validate keys in config
    df_values = get_default_args(compute_and_save)
    for k, v in [(k, v) for k, v in df_values.items() if k not in EMG_config]:
        print(f"Key {k} is missing from config. Using its default value: {v}")
    for k in [k for k in EMG_config if k not in df_values]:
        print(f"Config key {k} is not recognized")

    LFP_binPath = EMG_config.pop('LFP_binPath')

    compute_and_save(LFP_binPath, **EMG_config)


def compute_and_save(LFP_binPath, LFP_datatype=None, LFP_downsample=None,
                     LFP_chanList=None, EMGdata_savePath=None, overwrite=False,
                     sf=20.0, window_size=25.0, bandpass=None, bandstop=None,
                     gpass=1, gstop=20, ftype='butter'):

    # Manage default values:
    if bandpass is None:
        bandpass = [300, 600]
        print(f"Set bandpass={bandpass}")
    if bandstop is None:
        bandstop = [275, 625]
        print(f"Set bandstop={bandstop}")

    # Generate EMG metadata: save all the local variables in this function
    EMG_metadata = locals()

    # Get paths
    assert os.path.exists(LFP_binPath), "Data not found at {LFP_binPath}"
    if not EMGdata_savePath:
        EMGdatapath = Path(LFP_binPath.parent / (LFP_binPath.stem + ".derivedEMG.npy"))
    else:
        EMGdatapath = Path(EMGdata_savePath)
    EMGmetapath = get_EMGmetapath(EMGdatapath)

    # Do we compute the EMG?
    if os.path.exists(EMGdatapath) and os.path.exists(EMGmetapath):
        print(f"Found preexisting EMG files at {EMGdatapath}, {EMGmetapath}!")
        if overwrite:
            print('--> Recomputing EMG ( `overwrite` == True )')
        else:
            print('--> Exiting ( `overwite` == False )')
            return

    # Compute EMG
    print(f"Computing EMG from LFP:")
    print("Loading LFP for EMG computing")
    # Load LFP for channels of interest
    lfp, lfp_sf, chanLabels = load.loader_switch(
        LFP_binPath,
        datatype=LFP_datatype,
        chanList=LFP_chanList,
        downSample=LFP_downsample,
        tStart=None,
        # tEnd=None,  # Compute for whole recording
        tEnd=1000,  # Compute for whole recording
    )
    print(f"Using the following channels for EMG derivation (labels): "
          f"{' - '.join(chanLabels)}")
    print("Computing EMG from LFP")
    EMG_data = compute_EMG(
        lfp, lfp_sf,
        sf, window_size,
        bandpass, bandstop, gpass=gpass, gstop=gstop, ftype=ftype
    )

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


def get_default_args(func):
    import inspect
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        # if v.default is not inspect.Parameter.empty
    }
