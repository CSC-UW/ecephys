import os.path
from pathlib import Path

import numpy as np
import xarray as xr

from .emg import compute_emg
from . import load


def load_emg_npy(emg_data_path, tStart=None, tEnd=None, desired_length=None):
    emg_metadata_path = get_emg_metadata_path(emg_data_path)
    if not os.path.exists(emg_data_path) or not os.path.exists(emg_metadata_path):
        raise Exception(
            f"Couldn't find EMG files at {emg_data_path}, {emg_metadata_path}"
        )
    print(f"Load EMG at {emg_data_path}")
    emg_data = np.load(emg_data_path)
    emg_metadata = load.utils.load_yaml(emg_metadata_path)

    # Select time segment of interest for EMG
    sf = emg_metadata["sf"]
    if tStart is None:
        tStart = 0.0
    firstSamp = int(tStart * sf)
    if tEnd is None:
        lastSamp = emg_data.shape[1]
    else:
        lastSamp = int(tEnd * sf)
    if lastSamp > emg_data.shape[1]:
        raise Exception(
            f"tEnd={tEnd} beyond end of EMG. EMG was computed until"
            f" t={emg_data.shape[1]/sf}s"
        )
    print(
        f"Select timepoints between samples {firstSamp} and {lastSamp} out of"
        f" {emg_data.shape[1]}"
    )
    emg_data = emg_data[:, firstSamp : lastSamp + 1]

    if desired_length is not None:
        print(
            f"Resampling EMG slice from {emg_data.shape[1]} to "
            f"{desired_length} datapoints"
        )
        emg_data = load.resample.signal_resample(
            emg_data[0, :], desired_length=desired_length, method="numpy"
        ).reshape((1, desired_length))

    return emg_data, emg_metadata


def load_emg_netcdf(emg_data_path, tStart=None, tEnd=None, desired_length=None):
    emg = xr.open_dataset(emg_data_path).emg

    if tStart is None:
        tStart = emg.time.values.min()
    if tEnd is None:
        tEnd = emg.time.values.max()

    selected_data = emg.sel(time=slice(tStart, tEnd)).values

    if desired_length:
        print(
            f"Resampling EMG slice from {len(selected_data)} to {desired_length} datapoints"
        )
        selected_data = load.resample.signal_resample(
            selected_data, desired_length=desired_length, method="numpy"
        ).reshape((1, desired_length))

    return selected_data, emg.attrs


def load_emg(emg_data_path, tStart=None, tEnd=None, desired_length=None):
    """Load, slice and resample the EMG saved t `path`"""
    emg_data_path = Path(emg_data_path)
    supported_file_types = [".npy", ".nc"]
    assert path.suffix in supported_file_types, "Unsupported file type."

    if path.suffix == ".npy":
        return load_emg_npy(emg_data_path, tStart, tEnd, desired_length)

    if path.suffix == ".nc":
        return load_emg_netcdf(emg_data_path, tStart, tEnd, desired_length)


def run(emg_config):
    """Run `compute_and_save` from EMG config dictionary

    Args:
        emg_config: config dictionary containing the following keys:
            TODO
    """
    print(f"EMG config = {emg_config}, \n")

    # Validate keys in config
    df_values = get_default_args(compute_and_save)
    for k, v in [(k, v) for k, v in df_values.items() if k not in emg_config]:
        print(f"Key {k} is missing from config. Using its default value: {v}")
    for k in [k for k in emg_config if k not in df_values]:
        print(f"Config key {k} is not recognized")
    print("\n")

    lfp_bin_path = emg_config.pop("LFP_binPath")

    compute_and_save(lfp_bin_path, **emg_config)


def compute_and_save(
    LFP_binPath,
    LFP_datatype=None,
    LFP_downsample=None,
    LFP_chanList=None,
    LFP_tEnd=None,
    EMGdata_savePath=None,
    overwrite=False,
    sf=20.0,
    window_size=25.0,
    wp=None,
    ws=None,
    gpass=1,
    gstop=20,
    ftype="butter",
):

    # Manage default values:
    if wp is None:
        wp = [300, 600]
        print(f"Set wp (pass freq)={wp}")
    if ws is None:
        ws = [275, 625]
        print(f"Set ws (stop freq)={ws}")

    # Generate EMG metadata: save all the local variables in this function
    emg_metadata = locals()

    # Convert to Path after saving metadata
    LFP_binPath = Path(LFP_binPath)

    # Get paths
    assert os.path.exists(LFP_binPath), "Data not found at {LFP_binPath}"
    if not EMGdata_savePath:
        emg_data_path = Path(
            LFP_binPath.parent / (LFP_binPath.stem + ".derivedEMG.npy")
        )
    else:
        emg_data_path = Path(EMGdata_savePath)
    emg_metadata_path = get_emg_metadata_path(emg_data_path)

    # Do we compute the EMG?
    if os.path.exists(emg_data_path) and os.path.exists(emg_metadata_path):
        print(f"Found preexisting EMG files at {emg_data_path}, {emg_metadata_path}!")
        if overwrite:
            print("--> Recomputing EMG ( `overwrite` == True )")
        else:
            print("--> Exiting ( `overwite` == False )")
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
        tEnd=LFP_tEnd,
    )
    print(
        f"Using the following channels for EMG derivation (labels): "
        f"{' - '.join(chanLabels)}"
    )
    print("Computing EMG from LFP")
    EMG_data = compute_emg(
        lfp, lfp_sf, sf, window_size, wp, ws, gpass=gpass, gstop=gstop, ftype=ftype
    )

    # Save EMG
    print(f"Saving EMG metadata at {emg_metadata_path}")
    load.utils.save_yaml(emg_metadata_path, emg_metadata)
    print(f"Saving EMG data at {emg_data_path}")
    np.save(emg_data_path, EMG_data)

    return EMG_data, emg_metadata


def get_emg_metadata_path(emg_data_path):
    emg_data_path = Path(emg_data_path)
    metaName = emg_data_path.stem + ".meta.yml"
    return Path(emg_data_path.parent / metaName)


def get_default_args(func):
    import inspect

    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        # if v.default is not inspect.Parameter.empty
    }
