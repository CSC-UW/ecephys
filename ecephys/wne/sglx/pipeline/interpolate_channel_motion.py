from pathlib import Path
import probeinterface as pi
from ecephys.utils import siutils
import json
import logging

import numpy as np
from pyedflib import highlevel as edf
import scipy.interpolate
import xarray as xr

from ecephys import utils
from ecephys.wne import constants
from ecephys.wne.sglx import SGLXProject
from ecephys.wne.sglx import SGLXSubject
import ecephys.wne.utils as wne_utils

import pandas as pd
from docopt import docopt

import wisc_ecephys_tools as wet
from ecephys import wne
from ecephys import utils
from ecephys.wne.sglx.pipeline.sorting_pipeline import SpikeInterfaceSortingPipeline

from wisc_ecephys_tools.sortings import get_subject_probe_list
import shutil

def _prepare_motion_directory(
    project: SGLXProject,
    experiment: str,
    alias: str,
    sglx_subject: SGLXSubject,
    probe: str,
    sorting: str = "sorting",
):
    """Copy relevant data to `motion_best_estimate` sorting subdir.
    
    Pull either from `preprocessing` or `preprocessing.bak` 
    sorting subdirectory.
    """
    sorting_path = project.get_alias_subject_directory(
        experiment,
        alias,
        sglx_subject.name,
    )/f"{sorting}.{probe}"
    assert sorting_path.exists()

    motion_dir = sorting_path/"motion_best_estimate"
    if motion_dir.exists() and all([
        (motion_dir/fname).exists() for fname in [
            "motion_non_rigid_clean.npz",
            "opts.yaml",
        ]
    ]):
        return
        
    motion_dir.mkdir(exist_ok=True)

    prepro_path = sorting_path/"preprocessing"
    prepro_bak_path = sorting_path/"preprocessing.bak"
    assert(prepro_path.exists() or prepro_bak_path.exists())

    path = prepro_path if prepro_path.exists() else prepro_bak_path
    for src in path.glob("*"):
        tgt = motion_dir/src.name
        if not tgt.exists():
            shutil.copy(src, tgt)
        assert tgt.exists()
    
    if not (motion_dir/"opts.yaml").exists():
        src = prepro_path.parent/"opts.yaml"
        assert src.exists()
        tgt = motion_dir/src.name
        shutil.copy(src, tgt)


def _save_channel_motion(
    project: SGLXProject,
    experiment: str,
    alias: str,
    sglx_subject: SGLXSubject,
    probe: str,
    sorting: str = "sorting",
):
    """Load SI motion, interpolate per channel, and save as `channel_motion.nc`"""
    sorting_path = project.get_alias_subject_directory(
        experiment,
        alias,
        sglx_subject.name,
    )/f"{sorting}.{probe}"
    motion_dir = sorting_path/"motion_best_estimate"

    # Load motion info
    motion_path = motion_dir/"motion_non_rigid_clean.npz"
    npz = np.load(motion_path)
    motion = npz['motion']
    spatial_bins = npz['spatial_bins']
    temporal_bins = npz['temporal_bins']

    # Load channel depths from probe object with border channels removed
    probe_path = sorting_path/"preprocessed_si_probe.json"
    probe_group = pi.read_probeinterface(probe_path)
    assert len(probe_group.probes) == 1, "Expected to find only one probe"
    si_probe = probe_group.probes[0]
    channel_depths = si_probe.to_dataframe().sort_values(by="y")["y"].values

    # Load sampling rate
    segments_path = sorting_path/"segments.htsv"
    sampling_rate = utils.read_htsv(segments_path)["imSampRate"].values[0]

    # Sample2time
    sample2time = project.get_sample2time(
        sglx_subject.name,
        experiment=experiment,
        alias=alias,
        probe=probe,
        sorting=sorting,
    )

    channel_motion = siutils.interpolate_motion_per_channel(
        channel_depths,
        sampling_rate,
        motion,
        spatial_bins,
        temporal_bins,
        sample2time=sample2time
    )

    channel_motion.to_netcdf(
        motion_dir/"channel_motion.nc"
    )

    channel_motion.plot(figsize=(20, 10)).figure.savefig(
        motion_dir/"channel_motion.png"
    )


def do_sorting(
    project: SGLXProject,
    experiment: str,
    alias: str,
    sglx_subject: SGLXSubject,
    probe: str,
    sorting: str = "sorting",
):
    _prepare_motion_directory(
        project,
        experiment,
        alias,
        sglx_subject,
        probe,
        sorting,
    )

    _save_channel_motion(
        project,
        experiment,
        alias,
        sglx_subject,
        probe,
        sorting,
    )