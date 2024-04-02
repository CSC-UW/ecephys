import spikeinterface.core as si
from ecephys import utils
import pandas as pd
from ecephys.wne.sglx import utils as sglx_utils
import numpy as np
from offproj import core
from spikeinterface import preprocessing as sp
from spikeinterface.sortingcomponents import motion_interpolation
from ecephys.wne.sglx.pipeline.sorting_pipeline import SpikeInterfaceSortingPipeline
import dask.array
import xarray as xr
import zarr
from pathlib import Path
import numpy


DEFAULT_OPTS_NPX = {
    "bandpass_filt_min": 300,
    "bandpass_filt_max": 12000,
    "common_reference": "global",
    "gaussian_filt_max": 20,
    "decimation_factor": 100,
    "motion_correct": True,
}

DEFAULT_OPTS_TDT = {
    "bandpass_filt_min": 300,
    "bandpass_filt_max": 12000,
    "common_reference": "global",
    "gaussian_filt_max": 20,
    "decimation_factor": 100,
}


def preprocess_tdt_si_recording(
    si_rec: si.BaseRecording,
    time_vector: np.array,
    opts: dict = None,
):
    assert len(time_vector) == si_rec.get_num_samples()

    if opts is None:
        opts = DEFAULT_OPTS_TDT
    assert set(DEFAULT_OPTS_TDT.keys()) == set(opts.keys())

    print(f"Processing with opts: {opts}")

    pro_rec = si_rec
    bad_channel_ids, _ = sp.detect_bad_channels(pro_rec)
    print(bad_channel_ids)
    pro_rec = sp.bandpass_filter(pro_rec, opts["bandpass_filt_min"], opts["bandpass_filt_max"])
    pro_rec = sp.common_reference(pro_rec, reference=opts["common_reference"], operator="median")
    pro_rec = sp.interpolate_bad_channels(pro_rec, bad_channel_ids)
    pro_rec = sp.zscore(pro_rec, dtype='float32')
    pro_rec = sp.rectify(pro_rec)
    pro_rec = sp.gaussian_filter(pro_rec, freq_min=None, freq_max=opts["gaussian_filt_max"])

    pro_rec = sp.decimate(pro_rec, decimation_factor=opts["decimation_factor"])

    # Order by depth
    pro_rec = sp.depth_order(pro_rec)
    print("Done processing")

    # Assign decimated time vector so it's dumped in zarr group
    decimated_times = time_vector[::opts["decimation_factor"]]
    pro_rec.set_times(decimated_times, with_warning=False)

    return pro_rec


def preprocess_neuropixels_si_recording(
    si_rec: si.BaseRecording,
    time_vector: np.array,
    opts: dict = None,
    motion_npzfile: numpy.lib.npyio.NpzFile = None,
):
    assert len(time_vector) == si_rec.get_num_samples()

    if opts is None:
        opts = DEFAULT_OPTS_NPX
    assert set(DEFAULT_OPTS_NPX.keys()) == set(opts.keys())

    print(f"Processing with opts: {opts}")

    pro_rec = si_rec
    bad_channel_ids, _ = sp.detect_bad_channels(pro_rec)
    pro_rec = sp.bandpass_filter(pro_rec, opts["bandpass_filt_min"], opts["bandpass_filt_max"])
    pro_rec = sp.phase_shift(pro_rec)
    pro_rec = sp.common_reference(pro_rec, reference=opts["common_reference"], operator="median")
    pro_rec = sp.interpolate_bad_channels(pro_rec, bad_channel_ids)
    pro_rec = sp.zscore(pro_rec, dtype='float32')
    pro_rec = sp.rectify(pro_rec)
    pro_rec = sp.gaussian_filter(pro_rec, freq_min=None, freq_max=opts["gaussian_filt_max"])

    # After smoothing because this affects noise levels
    if motion_npzfile is not None and opts["motion_correct"]:
        motion = motion_npzfile["motion"]
        temporal_bins = motion_npzfile["temporal_bins"]
        spatial_bins = motion_npzfile["spatial_bins"]
        pro_rec = motion_interpolation.InterpolateMotionRecording(
            pro_rec,
            motion,
            temporal_bins,
            spatial_bins,
            direction=1,
            border_mode="remove_channels",
            spatial_interpolation_method="nearest",
            sigma_um=20.0,
            p=1,
            num_closest=3,
        )

    pro_rec = sp.decimate(pro_rec, decimation_factor=opts["decimation_factor"])

    # Order by depth
    pro_rec = sp.depth_order(pro_rec)
    print("Done processing")

    # Assign decimated time vector so it's dumped in zarr group
    decimated_times = time_vector[::opts["decimation_factor"]]
    pro_rec.set_times(decimated_times, with_warning=False)

    return pro_rec