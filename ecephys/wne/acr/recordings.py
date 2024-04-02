import tdt
import numpy as np
import tdt
import pickle
import pandas as pd
from ecephys import hypnogram
from ecephys.wne import constants
from ecephys.tdt.utils import set_probe_and_locations
import spikeinterface as si

try:
    import acr
    import acr.io

    HAS_ACR_PACKAGE = True
except ImportError:
    HAS_ACR_PACKAGE = False


def get_probe_segments_info(subject: str, probe: str, experiment: str) -> list[dict]:
    """Generate list of info dict for recording segments concatenated in a sorting."""

    assert (
        HAS_ACR_PACKAGE
    ), "Requires a couple functions from https://github.com/kortdriessen/acr/blob/main/acr/info_pipeline.py"

    recs_in_sorting, start_times, end_times = acr.units.get_time_info(subject, f"{experiment}-{probe}")
    info_times = acr.info_pipeline.subject_info_section(subject, "rec_times")

    segments_info = []

    for i, recording in enumerate(recs_in_sorting):
        end_time = end_times[i]

        t1 = 0  # This should be the case for all sortings so far...
        if (info_times[recording]["duration"] - end_time) < 0.0001:
            # Whole recording
            t2 = 0
        else:
            t2 = end_time
        # assert isinstance(t2, int), f"t2=={t2}. Expecting an integer"

        # t2 = 50  # TODO

        block_path = acr.io.acr_path(subject, recording)

        blk = tdt.read_block(
            block_path,
            store=probe,
            evtype=["streams"],
            t1=0,
            t2=t2,
            channel=[1],
        )  # single channel for fast loading

        num_channels = int(
            tdt.read_block(block_path, store=probe, evtype=["streams"], t1=0, t2=1).streams[probe].data.shape[0]
        )  # Short time range to get num chans

        segments_info.append(
            {
                "subject": subject,
                "experiment": experiment,
                "probe": probe,
                "recording": recording,
                "tdt_readblock_t1": t1,
                "tdt_readblock_t2": t2,
                "srate": float(blk.streams[probe].fs),
                "num_samples": int(blk.streams[probe].data.shape[0]),  # 0 because 1-d array when loading single chan
                "num_channels": num_channels,
                "start_datetime": start_times[i],
                "info_times": info_times[recording],
                "dtype": "float32",
                "tdt_block_path": acr.io.acr_path(subject, recording),
            }
        )

    return segments_info


def _get_relative_recording_start_time(segment_info, reference_timestamp):
    rec_start_timestamp = pd.Timestamp(segment_info["info_times"]["start"])
    return (rec_start_timestamp - reference_timestamp).total_seconds() + segment_info["tdt_readblock_t1"]


def _get_converted_recording_time_array(segment_info, reference_timestamp):
    """Get converted timestamps for a single recording."""
    raw_times = np.arange(segment_info["num_samples"]) / segment_info["srate"]
    return raw_times + _get_relative_recording_start_time(segment_info, reference_timestamp)


def get_converted_full_time_array(segments_info: list[dict]) -> np.array:
    """Get converted timestamps for all samples concatenated in a recording"""
    reference_timestamp = pd.Timestamp(segments_info[0]["info_times"]["start"])
    return np.concatenate(
        [
            _get_converted_recording_time_array(
                segment_info,
                reference_timestamp,
            )
            for segment_info in segments_info
        ]
    )


def load_raw_si_recording(segments_info: list[dict]) -> si.BaseRecording:
    rec = si.core.concatenate_recordings(
        [
            si.core.BinaryRecordingExtractor(
                info["bin_block_path"],
                info["srate"],
                dtype=info["dtype"],
                num_channels=info["num_channels"],
                channel_ids=[f"{info['probe']}-{i+1}" for i in range(info["num_channels"])],
                time_axis=1,
                gain_to_uV=1,
                offset_to_uV=0,  # TODO ???
                is_filtered=False,
            )
            for info in segments_info
        ]
    )
    rec = set_probe_and_locations(
        rec,
    )

    # I don't know why this is necessary, but "channel_name" should be a property
    # eg to be dumped in the zarr object
    rec.set_property("channel_name", rec.get_channel_ids(), ids=rec.get_channel_ids())

    return rec
