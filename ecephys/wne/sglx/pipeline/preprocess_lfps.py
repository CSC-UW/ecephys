# TODO: Handle dropping of duplicate timestamps across files?

import logging
from typing import Optional

from tqdm.auto import tqdm

from ecephys import sglxr
from ecephys import utils
from ecephys import xrsig
from ecephys.wne.sglx import SGLXProject
from ecephys.wne.sglx import SGLXSubject
import ecephys.wne.sglx.utils

logger = logging.getLogger(__name__)


def do_experiment(
    experiment: str,
    sglx_subject: SGLXSubject,
    sync_project: SGLXProject,
    dest_project: SGLXProject,
    opts_project: SGLXProject,
    chunk_duration: int = 300,  # Size of zarr chunks, in seconds
):
    lf_table = sglx_subject.get_lfp_bin_table(experiment, alias=alias)
    probes = lf_table["probe"].unique()
    for probe in probes:
        do_experiment_probe(
            experiment,
            probe,
            sglx_subject,
            sync_project,
            dest_project,
            opts_project,
            chunk_duration,
        )


def do_experiment_probe(
    experiment: str,
    probe: str,
    sglx_subject: SGLXSubject,
    sync_project: SGLXProject,
    dest_project: SGLXProject,
    opts_project: SGLXProject,
    chunk_duration: int = 300,  # Size of zarr chunks, in seconds
):
    """
    Parameters:
    -----------
    alias: str or None
        If None, do whole experiment


    IBL LFP pipeline
    -----
    1. 2-200Hz 3rd order butter bandpass, applied with filtfilt
    2. Destriping
        2.1 2 Hz 3rd order butter highpass, applied with filtfilt
        2.2 Dephasing
        2.3 Interpolation (inside brain)
        2.4 CAR
            2.4.1 Automatic gain control. What, exactly is this? Why do it?
            2.4.2 Median subtraction
    3. Decimation (10x)
        3.1 FIR anti-aliasing filter w/ phase correction

    WISC LFP pipeline
    -----
    1. Decimation (4x)
    2. Dephasing (adjusted for decimation)
    3. Interpolation

    Notes:
    -----
    - Time to load whole 2h file at once, all 384 channels: 6.5m
    - Time to load whole 2h file in 2**16 sample segments (~26s) with 2**10 sample overlap (~0.5s): ~15min
    - Time to dephase, interpolate, and decimate segments: 28m
    - Time to decimate, dephase, and interpolate: 7.5m.
        - Results are nearly identical. Resulting lfps are equivalent to within a picovolt.
        - Dephasing also continue to work just as well, provided your sample shifts account for the decimation.
    - Time to write float32 and float64 files are the same (~20s), but time to read float32 is twice as fast (5 vs 10s).
    - I have not yet tested whether using overlapping windows is truly necessary. It may not be, since the only filter here is the FIR antialiasing filter.
    - Note that the data here are NOT de-meaned.
    """
    opts = opts_project.load_experiment_subject_params(experiment, sglx_subject.name)
    bad_channels = opts["probes"][probe]["badChannels"]
    zarr_file = dest_project.get_experiment_subject_file(
        experiment, sglx_subject.name, f"{probe}.lf.zarr"
    )
    lf_table = sglx_subject.get_lfp_bin_table(experiment, probe=probe)
    for i, lfp_file in enumerate(tqdm(list(lf_table.itertuples()))):
        logger.info(f"Loading {lfp_file.path.name}...")
        lfp = sglxr.load_trigger(
            lfp_file.path,
            t0=lfp_file.expmtPrbAcqFirstTime,
        )
        logger.info(f"Converting to canonical timebase...")
        t2t = ecephys.wne.sglx.utils.get_time_synchronizer(
            sync_project, sglx_subject, experiment, binfile=lfp_file.path
        )
        lfp = lfp.assign_coords({"time": t2t(lfp["time"].values)})
        logger.info("Preprocessing...")
        lfp = xrsig.preprocess_neuropixels_ibl_style(lfp, bad_channels)
        lfp.name = "lfp"
        lfp.attrs = utils.drop_unserializeable(lfp.attrs)

        logger.info(f"Saving to: {zarr_file}")
        if i == 0:
            lfp = lfp.chunk(
                {"channel": lfp.channel.size, "time": int(lfp.fs * chunk_duration)}
            )  # If you choose to use the 'auto' chunksize setting, be warned: Different coords on the same dimension can be chunked differently.
            lfp.to_zarr(zarr_file, encoding={"lfp": {"dtype": "float32"}}, mode="w")
        else:
            lfp.to_zarr(zarr_file, append_dim="time")
    logger.info("Done preprocessing LFPs!")
