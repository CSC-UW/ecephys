import spikeinterface.extractors as se


def load_single_segment_sglx_recording(
    gate_dir, segment_idx, stream_id
):
    all_segments_rec = se.SpikeGLXRecordingExtractor(
        gate_dir,
        stream_id=stream_id,
    )
    assert isinstance(segment_idx, int)
    return all_segments_rec.select_segments(
        [segment_idx]
    )



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