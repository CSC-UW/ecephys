import logging

import numpy as np
import pandas as pd
from pandas.api import types
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


# TODO Delete??? Use project.get_kilosort_extractor??
# I don't think we should use KSLabel at all, rather keep the property as "unsorted"
# I don't think there is an issue with isi_violations_ratio
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


def add_anatomy_properties_to_extractor(
    extractor: se.KiloSortSortingExtractor, structs: pd.DataFrame
) -> se.KiloSortSortingExtractor:
    """
    Add a `structure` and `acronym` properties to each cluster indicating its anatomical region.

    Parameters
    ===========
    structure: The long structure name
    acronym: Abbreviated structure name
    hi: Upper boundary of the structure, in the same coordinates as the SI extractor's depth property
    lo: Lower boundary of the structure, in the same coordinates as the SI extractor's depth property

    Example structure table, as an HTSV file:
        structure	acronym	thickness	hi	lo
        Olfactory area / Basal forebrain / Dorsia tenia tecta	DTT	1192.4538258575196	1192.4538258575196	0.0
        Medial orbital cortex	MO	889.287598944591	2081.7414248021105	1192.4538258575196
        Prelimbic cortex / A32D + A32V	PreL	2506.174142480211	4587.915567282322	2081.7414248021105
        Secondary motor cortex	M2	2021.108179419525	6609.023746701847	4587.915567282322
        Out of brain	OOB	1050.9762532981529	7659.999999999999	6609.023746701847
    """

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
    extractor.set_annotation("structure_table", structs)
    return extractor


def get_dummy_structure_table():
    return pd.DataFrame(
        [{"structure": "Full probe", "acronym": "All", "lo": -np.Inf, "hi": np.Inf}]
    )


# TODO: Is this not used anymore? If not, why not?
def fix_isi_violations_ratio(
    extractor: se.KiloSortSortingExtractor,
) -> se.KiloSortSortingExtractor:
    """Kilosort computes isi_violations_ratio incorrectly, so fix it."""
    property_keys = extractor.get_property_keys()
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
    return extractor


def refine_clusters(si_obj, filters=None, include_nans=True):
    """Subselect clusters based on filters.

    Parameters:
    ===========
    si_obj: SI extractor or sorting object
    filters: dict
        Keys are the names of cluster properties.  Example properties include all columns in cluster_info.tsv, including quality metrics.
        Values are either 2-item tuples or sets.
        Tuples specify a range of allowable values for non-categorical numeric properties.
        Sets specify allowable values for categorical properties.
        For example, {"n_spikes": (2, np.Inf)} will load only clusters with 2 or more spikes.
        For example, {"quality": {"good", "mua"}} will load only clusters marked as such after curation and QMs.
    include_nans: bool, default True
        For several properties/metrics (including the cluster "group"/"quality" possibly set during curation),
        the value of the property may be np.NaN for some clusters. If True, we include those clusters (effectively
        filtering based on a property only when this property has a valid value)


    Returns:
    ========
    UnitsSelectionSorting

    Notes:
    ======
    SpikeInterface renames the 'group' columns in cluster_info.tsv to 'quality'.
    """
    if filters is None:
        filters = {}

    keep = np.ones_like(si_obj.get_unit_ids())
    for property, filter in filters.items():
        values = si_obj.get_property(property)
        if not property in si_obj.get_property_keys():
            logger.warning(
                f"Cluster property {property} not found. Unable to filter clusters based on {property}."
            )
            continue
        if isinstance(filter, tuple):
            lo, hi = filter
            assert types.is_numeric_dtype(
                np.array(filter)
            ), f"Expected a numeric dtype for values in tuple: `{filter}`. Specify values of interest as a set (rather than tuple) for non-numerical properties."
            assert types.is_numeric_dtype(
                values.dtype
            ), f"Cannot select a range of values for cluster property {property} with dtype {values.dtype}. Expected a numeric dtype."
            mask = np.logical_and(values >= lo, values <= hi)
        elif isinstance(filter, set):
            mask = np.isin(values, list(filter))
        else:
            raise ValueError(
                f"Cluster property {property} was provided as type {type(filter)}. Expected a tuple for selecting a range of numerical values, or a set for selecting categorical variables."
            )
        if include_nans:
            mask = pd.isna(values) | mask
        keep = keep & mask
        print(f"{property}: {filter} excludes {mask.size - mask.sum()} clusters.")

    print(
        f"{keep.size - keep.sum()}/{keep.size} clusters excluded by jointly applying filters."
    )
    clusterIDs = si_obj.get_unit_ids()[np.where(keep)]
    return si_obj.select_units(clusterIDs)


# TODO: Rename lowerBorder_imec to something non-neuropixel specific.
#   The only real requirement is that the units match those of the sorting object's depth property.
def add_structures_from_sharptrack(si_obj, sharptrack):
    """Use a SHARPTrack object to add a `structure` property to a SpikeInterface sorting object.

    Parameters:
    ===========
    sorting: UnitsSelectionSorting
        Must contain a 'depth' column
    sharptrack: ecephys.SHARPTrack
    """
    depths = si_obj.get_property("depth")
    structures = np.empty(depths.shape, dtype=object)
    for structure in sharptrack.structures.itertuples():
        lo = structure.lowerBorder_imec
        hi = structure.upperBorder_imec
        mask = (depths >= lo) & (depths <= hi)
        structures[np.where(mask)] = structure.acronym
    si_obj.set_property("structure", structures)
