import spikeinterface.extractors as se
from pathlib import Path

# TODO: Is this still used?
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
