import spikeinterface as si
import spikeinterface.extractors as se
from pathlib import Path
import scipy.interpolate
import numpy as np
import xarray as xr

import spikeinterface.extractors as se
from spikeinterface.core import concatenate_recordings, concatenate_sortings
from spikeinterface.core.waveform_tools import has_exceeding_spikes


def interpolate_motion_per_channel(
    channel_depths,
    sampling_rate,
    si_motion,
    si_spatial_bins,
    si_temporal_bins,
    sample2time = None,
) -> xr.DataArray:
    """
    Interpolate motion at each channel location and temporal bin.

    Args:
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
    
    Returns:
    xarray.DataArray with coordinates "channel_depth" (um), "sample_index" (int), and "time"
        if the `sample2time` conversion function is provided.

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
        channel_motions = np.empty(
            (len(channel_depths), len(temporal_bins))
        )
        for bin_ind, _ in enumerate(temporal_bins):
            # non rigid : interpolation channel motion for this temporal bin
            f = scipy.interpolate.interp1d(
                spatial_bins, si_motion[bin_ind, :], kind="linear", axis=0, bounds_error=False, fill_value="extrapolate"
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



def cut_and_combine_si_extractors(si_object, epochs_df, combine="concatenate"):
    assert {"start_frame", "end_frame", "state"}.issubset(epochs_df)
    assert len(epochs_df.state.unique()) == 1

    if not isinstance(si_object, (se.BaseSorting, se.BaseRecording)):
        raise ValueError(
            "Unrecognized datatype for si_object. "
            "Expected spikeinterface BaseSorting or BaseRecording."
        )

    frame_slice_kwargs = {}
    if isinstance(si_object, se.BaseSorting):
        # Disable redundant check_spike_frames in Sorting.frame_slice
        assert si_object.has_recording()
        if has_exceeding_spikes(si_object._recording, si_object):
            raise ValueError(
                "The sorting object has spikes exceeding the recording duration. You have to remove those spikes "
                "with the `spikeinterface.curation.remove_excess_spikes()` function"
            )
        frame_slice_kwargs = {'check_spike_frames': False}

    si_segments = []
    for epoch in epochs_df.itertuples():
        si_segments.append(
            si_object.frame_slice(
                start_frame=epoch.start_frame,
                end_frame=epoch.end_frame,
                **frame_slice_kwargs,
            )
        )

    if combine == "concatenate":

        if isinstance(si_object, se.BaseSorting):
            return si.concatenate_sortings(si_segments)
        elif isinstance(si_object, se.BaseRecording):
            return si.concatenate_recordings(si_segments)

    elif combine == "append":
        raise NotImplementedError

    assert False


def load_kilosort_bin_as_si_recording(
    ks_output_dir,
    fname="temp_wh.dat",
    si_probe=None,
):
    ks_output_dir = Path(ks_output_dir)
    recording_path = ks_output_dir / fname
    if not recording_path.exists():
        raise ValueError(
            f"Could not find bin file used for sorting at {recording_path}"
        )

    # Get recording.dat info from params.py
    d = {}
    with open(ks_output_dir / "params.py") as f:
        for line in f.readlines():
            (key, val) = line.rstrip("\n").split(" = ")
            d[key] = val
    d["sample_rate"] = float(d["sample_rate"])
    d["n_channels_dat"] = int(d["n_channels_dat"])
    d["dtype"] = str(d["dtype"].strip("'"))
    d["hp_filtered"] = bool(d["hp_filtered"])

    rec = se.BinaryRecordingExtractor(
        file_paths=recording_path,
        sampling_frequency=d["sample_rate"],
        num_chan=d["n_channels_dat"],
        dtype=d["dtype"],
        is_filtered=d["hp_filtered"],
    )
    assert d["hp_filtered"]

    if si_probe is not None:
        rec = rec.set_probe(si_probe)

    return rec
