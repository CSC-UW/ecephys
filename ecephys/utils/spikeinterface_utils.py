import spikeinterface.extractors as se
import spikeinterface as si

from probeinterface.io import read_spikeglx
from pathlib import Path


def load_postprocessed_bin_as_si_recording(
    ks_output_dir,
    fname="temp_wh.dat",
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

    # return set_probe_and_locations(rec)
    return rec


def load_si_waveform_extractor(
    ks_output_dir,
):

    sorting = se.read_kilosort(ks_output_dir)
    recording = load_postprocessed_bin_as_si_recording(ks_output_dir)

    waveform_dir = Path(ks_output_dir)/'waveforms'
    return extract_waveforms(
        recording,
        sorting,
        folder=waveform_dir,
        load_if_exists=True,
    )

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

# TODO
def run_si_metrics(si_sorting, opts):
    pass




# def set_probe_and_locations(recording, binpath):

#     idx, x, y = get_xy_coords(binpath)

#     locations = np.array([(x[i], y[i]) for i in range(len(idx))])
#     shape = "square"
#     shape_params = {"width": 8}

#     prb = Probe()
#     if "#SY0" in recording.channel_ids[-1]:
#         print("FOUND SY0")
#         ids = recording.channel_ids[:-1]  # Remove last (SY0)
#     else:
#         ids = recording.channel_ids
#     prb.set_contacts(locations[: len(ids), :], shapes=shape, shape_params=shape_params)
#     prb.set_contact_ids(ids)  # Must go after prb.set_contacts
#     prb.set_device_channel_indices(
#         np.arange(len(ids))
#     )  # Mandatory. I did as in recording.set_dummy_probe_from_locations
#     assert prb._contact_positions.shape[0] == len(
#         prb._contact_ids
#     )  # Shouldn't be needed

#     recording = recording.set_probe(prb)  # TODO: Use in_place=True ?

#     if any(["#SY0" in id for id in recording.channel_ids]):
#         assert False, "Did not expect to find SYNC channel"

#     return recording