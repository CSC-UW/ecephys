import spikeinterface.extractors as se
import spikeinterface as si

from probeinterface.io import read_spikeglx
from pathlib import Path


def load_kilosort_bin_as_si_recording(
    ks_output_dir,
    fname="temp_wh.dat",
    si_probe=None,
):
    ks_output_dir = Path(ks_output_dir)
    recording_path = ks_output_dir/fname
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
        is_filtered=d["hp_filtered"]
    )
    assert d["hp_filtered"]

    if si_probe is not None:
        rec = rec.set_probe(si_probe)

    return rec


def load_single_segment_sglx_recording(
    gate_dir, segment_idx, stream_id,
):
    all_segments_rec = se.SpikeGLXRecordingExtractor(
        gate_dir,
        stream_id=stream_id,
    )
    assert isinstance(segment_idx, int)
    return all_segments_rec.select_segments(
        [segment_idx]
    )