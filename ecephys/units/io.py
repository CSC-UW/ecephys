import logging
import numpy as np
import pandas as pd
import spikeinterface.extractors as se

logger = logging.getLogger(__name__)

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


def load_extractor(kilosort_output_dir) -> se.KiloSortSortingExtractor:
    """Get a KiloSort extractor, with various corrections made."""
    extractor = se.KiloSortSortingExtractor(kilosort_output_dir)
    property_keys = extractor.get_property_keys()

    # Fix isi_violations_ratio
    if all(
        np.isin(
            ["isi_violations_rate", "firing_rate", "isi_violations_ratio"],
            property_keys,
        )
    ):
        logger.info("Re-computing and overriding values for isi_violations_ratio.")
        extractor.set_property(
            "isi_violations_ratio",
            extractor.get_property("isi_violations_rate")
            / extractor.get_property("firing_rate"),
        )

    # KiloSort labels noise clusters as nan. Replacing with `noise` to match SI behavior.
    if "KSLabel" in property_keys:
        logger.info(
            "KiloSort labels noise clusters as nan. Replacing with `noise` to match SI behavior."
        )
        kslabel = extractor.get_property("KSLabel")
        kslabel[pd.isna(kslabel)] = "noise"
        extractor.set_property("KSLabel", kslabel)

    # If any clusters are uncurated, assign the KiloSort label
    if all(np.isin(["quality", "KSLabel"], property_keys)):
        quality = extractor.get_property("quality")
        uncurated = pd.isna(quality)
        if any(uncurated):
            logger.info(f"{uncurated.sum()} clusters are uncurated. Applying KSLabel.")
            kslabel = extractor.get_property("KSLabel")
            quality[uncurated] = kslabel[uncurated]
            extractor.set_property("quality", quality)

    return extractor


def add_structures_to_extractor(extractor, structs) -> se.KiloSortSortingExtractor:
    depths = extractor.get_property("depth")
    structures = np.empty(depths.shape, dtype=object)
    acronyms = np.empty(depths.shape, dtype=object)
    for structure in structs.itertuples():
        lo = structure.lo
        hi = structure.hi
        mask = (depths >= lo) & (depths <= hi)
        structures[np.where(mask)] = structure.structure
        acronyms[np.where(mask)] = structure.acronym
    extractor.set_property("structure", structures)
    extractor.set_property("acronym", acronyms)
    return extractor
