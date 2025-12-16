import shutil

import matplotlib.pyplot as plt
import numpy as np
import probeinterface as pi
import scipy.interpolate
import xarray as xr

import ecephys.utils
from ecephys.wne.sglx.project import SGLXProject
from ecephys.wne.sglx.subject import SGLXSubject


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
    sorting_path = (
        project.get_alias_subject_directory(
            experiment,
            alias,
            sglx_subject.name,
        )
        / f"{sorting}.{probe}"
    )
    assert sorting_path.exists()

    motion_dir = sorting_path / "motion_best_estimate"
    if motion_dir.exists() and all(
        [
            (motion_dir / fname).exists()
            for fname in [
                "motion_non_rigid_clean.npz",
                "opts.yaml",
            ]
        ]
    ):
        return

    motion_dir.mkdir(exist_ok=True)

    prepro_path = sorting_path / "preprocessing"
    prepro_bak_path = sorting_path / "preprocessing.bak"
    assert prepro_path.exists() or prepro_bak_path.exists()

    path = prepro_path if prepro_path.exists() else prepro_bak_path
    for src in path.glob("*"):
        tgt = motion_dir / src.name
        if not tgt.exists():
            shutil.copy(src, tgt)
        assert tgt.exists()

    if not (motion_dir / "opts.yaml").exists():
        src = prepro_path.parent / "opts.yaml"
        assert src.exists()
        tgt = motion_dir / src.name
        shutil.copy(src, tgt)


def _interpolate_motion_per_channel(
    channel_depths,
    sampling_rate,
    si_motion,
    si_spatial_bins,
    si_temporal_bins,
    sample2time,
) -> xr.DataArray:
    """
    Interpolate motion at each channel location and temporal bin.

    Parameters
    ----------
    channel_depths: np.array 1D
        Array-like of channel depths (y-axis location).
    sampling_rate: float
    si_motion: np.array 2D
        As returned by spikeinterface.estimate_motion
        motion.shape[0] equal temporal_bins.shape[0]
        motion.shape[1] equal 1 when "rigid" motion equal temporal_bins.shape[0] when "non-rigid"
    si_temporal_bins: np.array
        As returned by spikeinterface.estimate_motion
        Temporal bins in second (sorting time base)
    si_spatial_bins: np.array
        As returned by spikeinterface.estimate_motion
        Bins for non-rigid motion. If spatial_bins.sahpe[0] == 1 then rigid motion is used.
    sample2time: func
        As returned by SGLXProject.get_sample2time

    Returns
    -------
    channel_motion: xarray.DataArray
        da with dimensions "depth" and "time" dims & coordinates, and "sample_index" extra
        coordinates
    """
    temporal_bins = np.asarray(si_temporal_bins)
    spatial_bins = np.asarray(si_spatial_bins)
    channel_depths = np.asarray(channel_depths)
    if spatial_bins.shape[0] == 1:
        # same motion for all channels
        # No need to interpolate
        assert si_motion.shape[1] == 1
        channel_motions = np.tile(
            si_motion[:, 0],
            (len(channel_depths), 1),
        )
    else:
        channel_motions = np.empty((len(channel_depths), len(temporal_bins)))
        for bin_ind, _ in enumerate(temporal_bins):
            # non rigid : interpolation channel motion for this temporal bin
            f = scipy.interpolate.interp1d(
                spatial_bins,
                si_motion[bin_ind, :],
                kind="linear",
                axis=0,
                bounds_error=False,
                fill_value="extrapolate",
            )
            channel_motions[:, bin_ind] = f(channel_depths)
    sample_index = (temporal_bins * sampling_rate).astype(int)
    dims = ["depth", "time"]
    coords = {
        "depth": channel_depths,
        "sample_index": ("time", sample_index),
    }
    if sample2time is not None:
        try:
            times = sample2time(sample_index)
        except AssertionError:
            # Last temporal bin is beyond end of recording
            coords["sample_index"] = ("time", sample_index[:-1])
            channel_motions = channel_motions[:, :-1]
            times = sample2time(sample_index[:-1])
        coords["time"] = times
    return xr.DataArray(
        channel_motions,
        dims=dims,
        coords=coords,
        name="channel_motion",
        attrs=[
            ("depth", "um"),
            ("sample_index", "None"),
            ("time", "secs (sample2time)"),
        ],
    )


def _save_channel_motion(
    project: SGLXProject,
    experiment: str,
    alias: str,
    sglx_subject: SGLXSubject,
    probe: str,
    sorting: str = "sorting",
):
    """Load SI motion, interpolate per channel, and save as `channel_motion.nc`"""
    sorting_path = (
        project.get_alias_subject_directory(
            experiment,
            alias,
            sglx_subject.name,
        )
        / f"{sorting}.{probe}"
    )
    motion_dir = sorting_path / "motion_best_estimate"

    # Load motion info
    motion_path = motion_dir / "motion_non_rigid_clean.npz"
    npz = np.load(motion_path)
    motion = npz["motion"]
    spatial_bins = npz["spatial_bins"]
    temporal_bins = npz["temporal_bins"]

    # Load channel depths from probe object with border channels removed
    probe_path = sorting_path / "preprocessed_si_probe.json"
    probe_group = pi.read_probeinterface(probe_path)
    assert len(probe_group.probes) == 1, "Expected to find only one probe"
    si_probe = probe_group.probes[0]
    channel_depths = si_probe.to_dataframe().sort_values(by="y")["y"].values

    # Load sampling rate
    segments_path = sorting_path / "segments.htsv"
    sampling_rate = ecephys.utils.read_htsv(segments_path)["imSampRate"].values[0]

    # Sample2time
    sample2time = project.get_sample2time(
        sglx_subject.name,
        experiment=experiment,
        alias=alias,
        probe=probe,
        sorting=sorting,
    )

    channel_motion = _interpolate_motion_per_channel(
        channel_depths,
        sampling_rate,
        motion,
        spatial_bins,
        temporal_bins,
        sample2time=sample2time,
    )

    channel_motion.to_netcdf(motion_dir / "channel_motion.nc")

    channel_motion.plot(figsize=(20, 10)).figure.savefig(
        motion_dir / "channel_motion.png"
    )
    plt.close()


# TODO: This seems like it could have a better name. Maybe `do_probe()` or `interpolate_probe_motion()`?
def do_sorting(
    project: SGLXProject,
    experiment: str,
    alias: str,
    sglx_subject: SGLXSubject,
    probe: str,
    sorting: str = "sorting",  # TODO: ???
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
