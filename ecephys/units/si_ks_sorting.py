import logging
from typing import Callable, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics as skmetrics

import ephyviewer
import spikeinterface as si
import spikeinterface.extractors as se
from ecephys.units import ephyviewerutils, siutils
from ecephys.utils.misc import kway_sortednp_merge

logger = logging.getLogger(__name__)


class SpikeInterfaceKilosortSorting:
    def __init__(
        self,
        si_obj: Union[se.KiloSortSortingExtractor, si.UnitsSelectionSorting],
        sample2time: Optional[Callable] = None,
        hypnogram: Optional[pd.DataFrame] = None,
        structs: Optional[pd.DataFrame] = None,
    ):
        """
        This class is primarily used to map spike samples from one probe to spike times on another, since the SpikeInterface objects can not do this.

        Parameters:
        ===========
        si_obj:
            For example, from se.KiloSortSortingExtractor(sorting_dir)
        sample2time:
            Takes an array of spikeinterface sample numbers and maps them to timestamps. Ideally a vectorized function.
            If not provided, times will be according to the probe's sample clock (e.g. sample / fs)
        hypnogram:
            Hypnogram with 'start_time', 'end_time', 'duration' columns
            in the same timebase as the sample2time function, and
            `start_frame`, `end_frame` columns for each bout matching
            the spikeinterface sorting/recording frame ids.
        structs: pd.DataFrame
            Frame with `lo`, `hi`, `span`, `structure`, `acronym` fields.
            Cluster's structure/acronym assignation are added as properties, and the structure
            array is saved as `self.structs` attribute.
        """
        self.hypnogram: pd.DataFrame = hypnogram
        self.si_obj: se.KiloSortSortingExtractor = si_obj

        if structs is None:
            structs = pd.DataFrame(
                [
                    {
                        "structure": "Full probe",
                        "acronym": "All",
                        "lo": self.properties.depth.min(),
                        "hi": self.properties.depth.max(),
                    }
                ]
            )
        self.structs = structs
        self.si_obj = add_cluster_structures(self.si_obj, structs)

        # If no time mapping function is provided, just provide times according to this probe's sample clock.
        if sample2time is None:
            fs = si_obj.get_sampling_frequency()
            self.sample2time = lambda x: x / fs
        else:
            self.sample2time = sample2time

    def __repr__(self):
        return f"Wrapped {repr(self.si_obj)}"

    @property
    def properties(self) -> pd.DataFrame:
        """Return SI cluster properties as a DataFrame, with one row per cluster."""
        df = pd.DataFrame(self.si_obj._properties)
        df["cluster_id"] = self.si_obj.get_unit_ids()
        return df

    def plot_property_by_quality(
        self, property: str, ax: Optional[plt.Axes] = None
    ) -> plt.Axes:
        """Plot the distribution of property values (e.g. values of isi_violations_rate), separated by cluster quality.
        This can be helpful for seeing which properties correlate well with your labeling,
        or for identifying properties that might be used to effectively separate SUA from MUA clusters.
        """
        # These are useless properties. If asked for them, just return. This allows the user to loop through all properties without having to screen these out each time.
        if property in ["ch", "sh", "cluster_id"]:
            return

        # If no axis was provided, create our own.
        if ax is None:
            fig, ax = plt.subplots()

        df = self.properties
        # These properties are best visualized with a simple categorical bar plot
        if property in ["quality", "KSLabel"]:
            sns.histplot(data=df, x=property, ax=ax, color="k")
        # These properties are best visualized as stacked histograms, with hue shading by cluster quality
        elif property in [
            "ContamPct",
            "Amplitude",
            "amp",
            "isi_violations_rate",
            "amplitude_cutoff",
            "n_spikes",
            "isi_violations_count",
            "structure",
        ]:
            sns.histplot(data=df, x=property, hue="quality", multiple="stack", ax=ax)
        # Firing rate requires a specific binwidth to display sensibly.
        elif property in ["firing_rate", "fr"]:
            sns.histplot(
                data=df,
                x=property,
                hue="quality",
                multiple="stack",
                binwidth=0.1,
                ax=ax,
            )
        # Ignore spurious clusters with > 3% ISI violations.
        elif property == "isi_violations_ratio":
            sns.histplot(data=df, x=property, hue="quality", multiple="stack", ax=ax)
            ax.set_xlim(0.0, 0.03)
        # SNR requires a specific binwidth to display sensibly.
        elif property == "snr":
            sns.histplot(
                data=df, x=property, hue="quality", multiple="stack", binwidth=2, ax=ax
            )
        # Presence ratio requires a specific binwidth to display sensibly.
        elif property == "presence_ratio":
            sns.histplot(
                data=df,
                x=property,
                hue="quality",
                multiple="stack",
                binwidth=0.01,
                ax=ax,
            )
        # Depth requires a specific binwidth to display sensibly, and should be plotted along y axis
        elif property == "depth":
            sns.histplot(
                data=df,
                y=property,
                hue="quality",
                multiple="stack",
                binwidth=100,
                ax=ax,
            )
        else:
            raise ValueError(f"Unrecognized property: {property}")

        return ax

    def plot_property_by_depth(
        self, property: str, ax: Optional[plt.Axes] = None
    ) -> plt.Axes:
        """Plot the distribution of property values (e.g. values of isi_violations_rate), by the cluster's depth.
        This can be helpful for seeing whether different anatomical regions have very different firing properties.
        """
        # These are useless properties. If asked for them, just return. This allows the user to loop through all properties without having to screen these out each time.
        if property in ["ch", "sh", "cluster_id", "quality", "KSLabel", "depth"]:
            return

        # If no axis was provided, create our own.
        if ax is None:
            fig, ax = plt.subplots()

        df = self.properties
        if property in [
            "ContamPct",
            "Amplitude",
            "amp",
            "isi_violations_rate",
            "isi_violations_ratio",
            "amplitude_cutoff",
            "n_spikes",
            "isi_violations_count",
            "firing_rate",
            "fr",
            "snr",
            "presence_ratio",
        ]:
            sns.scatterplot(data=df, x=property, y="depth", hue="quality", ax=ax)
        # Provide per-channel resolution when visualizing structure.
        elif property == "structure":
            sns.histplot(data=df, y="depth", hue=property, binwidth=20)
        else:
            raise ValueError(f"Unrecognized property: {property}")

        return ax

    def plot_confusion(self, ax: Optional[plt.Axes] = None) -> plt.Axes:
        """Plot a confusion matrix to compare manual curation labels with automatically assigned KS labels."""
        if ax is None:
            fig, ax = plt.subplots()

        df = self.properties
        skmetrics.ConfusionMatrixDisplay.from_predictions(
            df["quality"], df["KSLabel"], ax=ax
        )
        return ax

    def refine_clusters(self, filters: dict, include_nans: bool = True):
        """Refine clusters, and conveniently wrap the result, so that the user doesn't have to."""
        new_obj = siutils.refine_clusters(self.si_obj, filters, include_nans=include_nans)
        return self.__class__(
            new_obj, self.sample2time, hypnogram=self.hypnogram, structs=self.structs
        )

    def select_clusters(self, clusterIDs):
        """Select clusters, and conveniently wrap the result, so that the user doesn't have to."""
        new_obj = self.si_obj.select_units(clusterIDs)
        return self.__class__(
            new_obj, self.sample2time, hypnogram=self.hypnogram, structs=self.structs
        )

    def _get_aggregate_train(self, property_column, property_value, as_sample_indices=False) -> np.array:
        mask = self.properties[property_column] == property_value
        tgt_clusters = self.properties[mask].cluster_id.values
        return kway_sortednp_merge(
            [
                train
                for train in self.get_trains_by_cluster_ids(
                    cluster_ids=tgt_clusters,
                    as_sample_indices=as_sample_indices,
                ).values()
            ]
        )

    def get_trains_by_cluster_ids(self, cluster_ids=None, as_sample_indices=False) -> dict:
        """Get spike trains for a list of clusters (default all)."""
        if cluster_ids is None:
            cluster_ids = self.si_obj.get_unit_ids()
        if as_sample_indices:
            func = lambda x: x
        else:
            func = self.sample2time
        return {
            id: func(self.si_obj.get_unit_spike_train(id))
            for id in cluster_ids
        }  # Getting spike trains cluster-by-cluster is MUCH faster than getting them all together.

    def get_trains(self, by="cluster_id", tgt_values=None, as_sample_indices=False) -> dict:

        if by == "cluster_id":
            return self.get_trains_by_cluster_ids(cluster_ids=tgt_values, as_sample_indices=as_sample_indices)

        if tgt_values is None:
            tgt_values = self.properties[by].unique()

        return {v: self._get_aggregate_train(by, v, as_sample_indices=as_sample_indices) for v in tgt_values}

    def add_ephyviewer_spiketrain_views(
        self, window, by: str = "depth", tgt_struct_acronyms: list[str] = None,
    ):

        all_structures = self.structs.sort_values(
            by="lo", ascending=False
        ).acronym.values  # Descending depths

        if tgt_struct_acronyms is None:
            tgt_struct_acronyms = all_structures
        else:
            assert all([s in all_structures for s in tgt_struct_acronyms])

        for tgt_struct_acronym in tgt_struct_acronyms:
            window = ephyviewerutils.add_spiketrainviewer_to_window(
                window, self, by=by, tgt_struct_acronym=tgt_struct_acronym, probe=None
            )

        return window

    def add_ephyviewer_hypnogram_view(self, window):
        if self.hypnogram is not None:
            window = ephyviewerutils.add_hypnogram_to_window(window, self.hypnogram)
        return window

    def plot_interactive_ephyviewer_raster(
        self, by: str = "depth", tgt_struct_acronyms: list[str] = None, 
    ):

        app = ephyviewer.mkQApp()
        window = ephyviewer.MainViewer(
            debug=True, show_auto_scale=True, global_xsize_zoom=True
        )

        window = self.add_ephyviewer_hypnogram_view(window)
        window = self.add_ephyviewer_spiketrain_views(window, by=by, tgt_struct_acronyms=tgt_struct_acronyms)

        window.show()
        app.exec()


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


# DEPRECATED: Because we should never be using KSLabel.
def fix_noise_cluster_labels(
    extractor: se.KiloSortSortingExtractor,
) -> se.KiloSortSortingExtractor:
    """Although cluster_KSLabel.tsv never contains nan, when loaded by SpikeInterface, KSLabel can be nan. These are noise clusters, so we relabel as `noise` to match SI behavior."""
    property_keys = extractor.get_property_keys()
    if "KSLabel" in property_keys:
        logger.info(
            "KiloSort labels noise clusters at nan. Replacing with `noise` to match SI behavior."
        )
        kslabel = extractor.get_property("KSLabel")
        kslabel[pd.isna(kslabel)] = "noise"
        extractor.set_property("KSLabel", kslabel)
    return extractor


# DEPRECATED: Because we should never be using KSLabel.
def fix_uncurated_cluster_labels(
    extractor: se.KiloSortSortingExtractor,
) -> se.KiloSortSortingExtractor:
    """If clusters are uncurated, fall back on the label that Kilosort assigned to the cluster."""
    property_keys = extractor.get_property_keys()
    if all(np.isin(["quality", "KSLabel"], property_keys)):
        quality = extractor.get_property("quality")
        # If no clusters are curated, quality will be all NaN with type float, instead of type object
        # Attempts to assign strings (e.g. "good") to this Series will result in a Type Error
        # Cast to object to avoid this, and so that the type of the property on the extractor is consistent
        quality = quality.astype("object")
        uncurated = pd.isna(quality)
        if any(uncurated):
            print(f"{uncurated.sum()} clusters are uncurated. Applying KSLabel.")
            kslabel = extractor.get_property("KSLabel")
            quality[uncurated] = kslabel[uncurated]
            extractor.set_property("quality", quality)
    return extractor


def add_cluster_structures(
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
    return extractor
