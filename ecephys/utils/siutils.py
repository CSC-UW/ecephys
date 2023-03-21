from spikeinterface.core import concatenate_sortings, concatenate_recordings
import spikeinterface.extractors as se
from pathlib import Path


def cut_and_combine_si_extractors(si_object, epochs_df, combine="concatenate"):
    assert {"start_frame", "end_frame", "state"}.issubset(epochs_df)
    assert len(epochs_df.state.unique()) == 1

    if not isinstance(si_object, (se.BaseSorting, se.BaseRecording)):
        raise ValueError(
            "Unrecognized datatype for si_object. "
            "Expected spikeinterface BaseSorting or BaseRecording."
        )

    si_segments = []
    for epoch in epochs_df.itertuples():
        si_segments.append(
            si_object.frame_slice(
                start_frame=epoch.start_frame,
                end_frame=epoch.end_frame
            )
        )
    
    if combine == "concatenate":

        if isinstance(si_object, se.BaseSorting):
            return concatenate_sortings(si_segments)
        elif isinstance(si_object, se.BaseRecording):
            return concatenate_recordings(si_segments)

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
