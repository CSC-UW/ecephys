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
import ecephys
from tqdm import tqdm
import itertools

try:
    import on_off_detection
except ImportError:
    _has_on_off_detection = False
else:
    _has_on_off_detection = True

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

        self._structs = structs
        self.si_obj = add_cluster_structures(self.si_obj, self.structs)

        # If no time mapping function is provided, just provide times according to this probe's sample clock.
        self._sample2time = sample2time

        # Cache cluster's whole-recording spike train in sample_idx
        self._strains_by_cluster_id = {}

    def __repr__(self):
        return f"Wrapped {repr(self.si_obj)}"

    @property
    def sample2time(self):
        if self._sample2time is None:
            fs = self.si_obj.get_sampling_frequency()
            return lambda x: x / fs
        else:
            return self._sample2time

    @property
    def structs(self) -> pd.DataFrame:
        if self._structs is None:
            return pd.DataFrame(
                [
                    {
                        "structure": "Full probe",
                        "acronym": "All",
                        "lo": self.properties.depth.min(),
                        "hi": self.properties.depth.max(),
                    }
                ]
            )
        return self._structs

    @property
    def properties(self) -> pd.DataFrame:
        """Return SI cluster properties as a DataFrame, with one row per cluster."""
        df = pd.DataFrame(self.si_obj._properties)
        df["cluster_id"] = self.si_obj.get_unit_ids()
        return df

    @property
    def has_anatomy(self):
        return self._structs is not None

    @property
    def has_hypnogram(self):
        return self.hypnogram is not None

    @property
    def has_sample2time(self):
        return self._sample2time is not None

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
        new_obj = siutils.refine_clusters(
            self.si_obj, filters, include_nans=include_nans
        )
        return self.__class__(
            new_obj, self.sample2time, hypnogram=self.hypnogram, structs=self.structs
        )

    def select_clusters(self, clusterIDs):
        """Select clusters, and conveniently wrap the result, so that the user doesn't have to."""
        new_obj = self.si_obj.select_units(clusterIDs)
        return self.__class__(
            new_obj, self.sample2time, hypnogram=self.hypnogram, structs=self.structs
        )
    
    def _get_cluster_strain(self, cluster_id):
        if cluster_id not in self._strains_by_cluster_id:
            self._strains_by_cluster_id[cluster_id] = self.si_obj.get_unit_spike_train(cluster_id)
        return self._strains_by_cluster_id[cluster_id]

    def _get_aggregate_train(
        self, property_column, property_value, as_sample_indices=False
    ) -> np.array:
        mask = self.properties[property_column] == property_value
        tgt_clusters = self.properties[mask].cluster_id.values
        return kway_sortednp_merge(
            [
                train
                for train in self.get_trains_by_cluster_ids(
                    cluster_ids=tgt_clusters,
                    as_sample_indices=as_sample_indices,
                    verbose=False,
                ).values()
            ]
        )

    def get_trains_by_cluster_ids(
        self,
        cluster_ids=None,
        as_sample_indices=False,
        verbose=True,
    ) -> dict:
        """Get spike trains for a list of clusters (default all)."""
        if cluster_ids is None:
            cluster_ids = self.si_obj.get_unit_ids()

        if as_sample_indices:
            func = lambda x: x
        else:
            func = self.sample2time

        if verbose:
            iterator = tqdm(cluster_ids, desc="Loading trains by clusters: ")
        else:
            iterator = cluster_ids

        return {
            id: func(self._get_cluster_strain(id)) for id in iterator
        }  # Getting spike trains cluster-by-cluster is MUCH faster than getting them all together.

    def get_trains(
        self,
        by="cluster_id",
        tgt_values=None,
        as_sample_indices=False,
        verbose=True,
    ) -> dict:

        if by == "cluster_id":
            return self.get_trains_by_cluster_ids(
                cluster_ids=tgt_values,
                as_sample_indices=as_sample_indices,
                verbose=verbose,
            )

        if tgt_values is None:
            tgt_values = self.properties[by].unique()

        if verbose:
            iterator = tqdm(tgt_values, desc="Loading trains by `{by}`: ")
        else:
            iterator = tgt_values

        return {
            v: self._get_aggregate_train(by, v, as_sample_indices=as_sample_indices)
            for v in iterator
        }

    def add_ephyviewer_spiketrain_views(
        self,
        window,
        by: str = "depth",
        tgt_struct_acronyms: list[str] = None,
        group_by_structure: bool = True,
    ):

        if group_by_structure:

            all_structures = self.structs.sort_values(
                by="lo", ascending=False
            ).acronym.values  # Descending depths

            if tgt_struct_acronyms is None:
                tgt_struct_acronyms = all_structures
            else:
                assert all([s in all_structures for s in tgt_struct_acronyms])

            for tgt_struct_acronym in tgt_struct_acronyms:
                window = ephyviewerutils.add_spiketrainviewer_to_window(
                    window,
                    self,
                    by=by,
                    tgt_struct_acronym=tgt_struct_acronym,
                    probe=None,
                )

        else:
            # Full probe
            window = ephyviewerutils.add_spiketrainviewer_to_window(
                window, self, by=by, tgt_struct_acronym=None, probe=None
            )

        return window

    def add_ephyviewer_hypnogram_view(self, window):
        if self.hypnogram is not None:
            window = ephyviewerutils.add_epochviewer_to_window(
                window,
                self.hypnogram,
                view_name="Hypnogram",
                name_column="state",
                color_by_name=ecephys.plot.state_colors,
            )
        return window

    def plot_interactive_ephyviewer_raster(
        self,
        by: str = "depth",
        tgt_struct_acronyms: list[str] = None,
    ):

        app = ephyviewer.mkQApp()
        window = ephyviewer.MainViewer(
            debug=True, show_auto_scale=True, global_xsize_zoom=True
        )

        window = self.add_ephyviewer_hypnogram_view(window)
        window = self.add_ephyviewer_spiketrain_views(
            window, by=by, tgt_struct_acronyms=tgt_struct_acronyms
        )

        window.show()
        app.exec()

    def run_off_detection(
        self,
        tgt_states=None,
        split_by_state=True,
        on_off_method="hmmem",
        on_off_params=None,
        spatial_detection=False,
        spatial_params=None,
        split_by_structure=False,
        tgt_structure_acronyms=None,
        n_jobs=10,
        min_sum_fr=30,
    ):
        """Run global/local OFF detection.

        Parameters:
        ===========
        tgt_states: list[str]
            List of hypnogram states that are cut and concatenated.  "NoData"
            epochs are excluded. Epochs not fully comprised within recording
            start/end are
            excluded.
        split_by_state: bool
            If True, we run OFF detection separately for each of the hypnogram states of interest
        on_off_method: str
            Method for OFF detection (default "hmmem")
        on_off_params: str
            Params for OFF detection
        spatial_detection: bool
            If True, perform spatial OFF detection
            (on_off_detection.SpatialOffModel). Otherwise, perform global OFF
            detection (on_off_detection.OnOffModel).  If False, `spatial_params`
            are ignored. (default False)
        spatial_params: dict
            Params for spatial aggregation of OFF periods. Must be None if
            `spatial_detection` is False.  split_by_structure: bool If True, off
            (spatial) detection is performed separately for all target
            structures. (default False)
        tgt_structure_acronyms: list[str]
            List of structure acronyms for which (spatial/global) OFF detection
            is performed. If `split_by_structure` is False, a single pass of
            (global/spatial) off detection is performed, for all structures at
            once. By default, all structures are included.
        n_jobs: int
            Parallelize across windows. Spatial off detection only.
        min_sum_fr: float
            If the cumulative firing rate for the region/sorting of interest 
            is below this value, do nothing (default 30)

        
        Return:
        =======
        pd.DataFrame:
            Off period frame with "state" (all "off"), "start_time", "end_time", "duration" columns,
            and possibly "window_depths", ... columns for spatial Off detection
        """

        if not _has_on_off_detection:
            raise ImportError(
                "Please install `on_off_detection` package from https://github.com/csc-uw/on_off_detection"
            )
        if not self.has_hypnogram:
            raise ValueError(
                "A hypnogram is required for off detection (to exclude 'NoData' epochs)."
            )
        if not spatial_detection:
            assert (
                spatial_params is None
            ), f"Set `spatial_params=None` if `spatial_detection` is False."

        print(
            f"Running ON/OFF detection. Cutting/concatenating the following hypnogram states: {tgt_states}"
        )

        # Get trains from structures of interest
        all_structure_acronyms = [s for s in self.structs.acronym.unique()]
        if tgt_structure_acronyms is None:
            tgt_structure_acronyms = all_structure_acronyms
        assert all([s in all_structure_acronyms for s in tgt_structure_acronyms]), (
            f"Invalid value in `tgt_structure_acronyms={tgt_structure_acronyms}`. "
            f"Available structures: {all_structure_acronyms}"
        )
        properties = self.properties[
            self.properties.acronym.isin(tgt_structure_acronyms)
        ]
        all_trains = self.get_trains_by_cluster_ids(
            cluster_ids=properties.cluster_id.values, verbose=True
        )

        # Get requested subset of epochs and check they have actual spiking activity
        # Remove "NoData"
        all_allowed_states = [s for s in self.hypnogram.state.unique() if s != "NoData"]
        if tgt_states is None:
            tgt_states = all_allowed_states
        assert all(
            [s in all_allowed_states for s in tgt_states]
        ), f"Invalid value in `tgt_states={tgt_states}`. Available states: {all_allowed_states}"
        mask = self.hypnogram.state.isin(tgt_states)
        # Remove epochs starting/ending before/after start/end of recording (avoid spurious OFF)
        first_spike_t = min([min(t) for t in all_trains.values()])
        last_spike_t = max([max(t) for t in all_trains.values()])
        mask = (
            mask
            & (self.hypnogram["start_time"] >= first_spike_t)
            & (self.hypnogram["end_time"] <= last_spike_t)
        )
        hypnogram = self.hypnogram[mask]

        # Iterate on structures of interest
        if split_by_structure:
            structures_to_aggregate = [[s] for s in tgt_structure_acronyms]
        else:
            structures_to_aggregate = [tgt_structure_acronyms]
        
        # Iterate on states of interest
        if split_by_state:
            states_to_aggregate = [[s] for s in hypnogram.state.unique()]
        else:
            states_to_aggregate = [hypnogram.state.unique()]

        all_structures_dfs = []
        for structures, states in itertools.product(structures_to_aggregate, states_to_aggregate):
            tgt_properties = properties[properties.acronym.isin(structures)]
            tgt_trains = [
                all_trains[row.cluster_id] for row in tgt_properties.itertuples()
            ]
            tgt_cluster_ids = [row.cluster_id for row in tgt_properties.itertuples()]
            tgt_depths = [row.depth for row in tgt_properties.itertuples()]

            sumFR = tgt_properties["fr"].sum()
            if sumFR <= min_sum_fr:
                print(
                    f"Too few spikes (sumFR={sumFR}Hz) in the following structures: {structures}. Passing ON/OFF detection"
                )
                continue
            else:
                print(
                    f"Running ON/OFF detection for structures {structures}, states {states}.\n"
                    f"N={len(tgt_trains)}units, sumFR={sumFR}Hz"
                )

            try:
                if not spatial_detection:
                    df = on_off_detection.OnOffModel(
                        tgt_trains,
                        None,
                        cluster_ids=tgt_cluster_ids,
                        method=on_off_method,
                        params=on_off_params,
                        bouts_df=hypnogram[hypnogram["state"].isin(states)],
                    ).run()
                else:
                    df = on_off_detection.SpatialOffModel(
                        tgt_trains,
                        tgt_depths,
                        None,
                        cluster_ids=tgt_cluster_ids,
                        on_off_method=on_off_method,
                        on_off_params=on_off_params,
                        spatial_params=spatial_params,
                        bouts_df=hypnogram[hypnogram["state"].isin(states)],
                        n_jobs=n_jobs,
                    ).run()
            except on_off_detection.ALL_METHOD_EXCEPTIONS as e:
                print(f"\n\nException for structures {structures}: {e}\n\n Passing.\n")
                continue
            df["structures"] = [structures] * len(df)
            df["states"] = [states] * len(df)

            if len(df):
                all_structures_dfs.append(df[df["state"] == "off"])

        if not len(all_structures_dfs):
            return pd.DataFrame()

        return pd.concat(all_structures_dfs).reset_index(drop=True)


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
