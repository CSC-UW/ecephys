import spikeinterface.extractors as se


# TODO: Is this used, and for what, and by whom? Does it belong in a project repo?
def get_sorting_info(ks_dir):
    # Read params.py
    d = {}
    with open(ks_dir / "params.py") as f:
        for line in f.readlines():
            (key, val) = line.rstrip("\n").split(" = ")
            d[key] = val
    d["sample_rate"] = float(d["sample_rate"])
    d["n_channels_dat"] = int(d["n_channels_dat"])
    d["dtype"] = str(d["dtype"].strip("'"))
    d["hp_filtered"] = bool(d["hp_filtered"])
    # duration
    tmp_extr = se.BinaryRecordingExtractor(
        file_paths=ks_dir / "temp_wh.dat",
        sampling_frequency=d["sample_rate"],
        num_chan=d["n_channels_dat"],
        dtype=d["dtype"],
    )
    d["duration"] = tmp_extr.get_num_frames() / tmp_extr.get_sampling_frequency()
    return d
