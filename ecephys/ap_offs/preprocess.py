import numpy as np
import spikeinterface.core as si
import spikeinterface.preprocessing as sp
from numpy.lib import npyio
from spikeinterface.sortingcomponents import motion_interpolation

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
    pro_rec = sp.bandpass_filter(
        pro_rec, opts["bandpass_filt_min"], opts["bandpass_filt_max"]
    )
    pro_rec = sp.common_reference(
        pro_rec, reference=opts["common_reference"], operator="median"
    )
    pro_rec = sp.interpolate_bad_channels(pro_rec, bad_channel_ids)
    pro_rec = sp.zscore(pro_rec, dtype="float32")
    pro_rec = sp.rectify(pro_rec)
    pro_rec = sp.gaussian_filter(
        pro_rec, freq_min=None, freq_max=opts["gaussian_filt_max"]
    )

    pro_rec = sp.decimate(pro_rec, decimation_factor=opts["decimation_factor"])

    # Order by depth
    pro_rec = sp.depth_order(pro_rec)
    print("Done processing")

    # Assign decimated time vector so it's dumped in zarr group
    decimated_times = time_vector[:: opts["decimation_factor"]]
    pro_rec.set_times(decimated_times, with_warning=False)

    return pro_rec


def preprocess_neuropixels_si_recording(
    recording: si.BaseRecording,
    times: np.ndarray,
    opts: dict = None,
    motion_npz: npyio.NpzFile = None,
) -> si.BaseRecording:
    assert len(times) == recording.get_num_samples(), (
        f"Time vector length {len(times)} does not match recording length {recording.get_num_samples()}"
    )

    if opts is None:
        opts = DEFAULT_OPTS_NPX
    assert set(DEFAULT_OPTS_NPX.keys()) == set(opts.keys())

    print(f"Processing with opts: {opts}")

    bad_channel_ids, _ = sp.detect_bad_channels(recording)
    recording = sp.bandpass_filter(
        recording, opts["bandpass_filt_min"], opts["bandpass_filt_max"]
    )
    recording = sp.phase_shift(recording)
    recording = sp.common_reference(
        recording, reference=opts["common_reference"], operator="median"
    )
    recording = sp.interpolate_bad_channels(recording, bad_channel_ids)
    recording = sp.zscore(recording, dtype="float32")
    recording = sp.rectify(recording)
    recording = sp.gaussian_filter(
        recording, freq_min=None, freq_max=opts["gaussian_filt_max"]
    )
    if motion_npz is not None and opts["motion_correct"]:
        motion = motion_npz["motion"]
        temporal_bins = motion_npz["temporal_bins"]
        spatial_bins = motion_npz["spatial_bins"]
        recording = motion_interpolation.InterpolateMotionRecording(
            recording,
            motion,
            temporal_bins,
            spatial_bins,
            direction=1,
            border_mode="remove_channels",
            spatial_interpolation_method="nearest",
            sigma_um=20.0,
            p=1,
            num_closest=3,
        )  # This must follow smoothing, because this affects noise levels
    recording = sp.decimate(recording, decimation_factor=opts["decimation_factor"])
    recording = sp.depth_order(recording)

    # Assign decimated time vector so it's dumped in zarr group
    decimated_times = times[:: opts["decimation_factor"]]
    recording.set_times(decimated_times, with_warning=False)

    return recording
