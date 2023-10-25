import logging
from typing import Callable, Optional, Union, Any
import warnings

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
import sklearn.metrics as skmetrics
import spikeinterface as si
import spikeinterface.extractors as se
from tqdm import tqdm

from ecephys import hypnogram
import ecephys.plot
from ecephys.units import cluster_trains
from ecephys.units import dtypes
from ecephys.units import siutils
import ecephys.utils

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
        cache: Optional[dtypes.ClusterTrains_Samples] = None,
    ):
        """
        This class is primarily used to map spike samples (and therfore times) from one probe to spike times on another, since the SpikeInterface objects can not do this.

        Parameters:
        ===========
        si_obj:
            For example, from se.KiloSortSortingExtractor(sorting_dir)
        sample2time:
            Takes an array of spikeinterface sample numbers and maps them to timestamps. Ideally a vectorized function.
            If not provided, times will be according to the probe's sample clock (e.g. sample / fs)
        cache: ClusterTrains_Samples:
            Optionally initialize the spike train cache. We cache each cluster's whole-recording spike train (as sample indices, not seconds). This is used to improve raster plot performance.
        """
        self.si_obj: se.KiloSortSortingExtractor = si_obj

        # If no time mapping function is provided, just provide times according to this probe's sample clock.
        self._sample2time = sample2time

        # We cache each cluster's whole-recording spike train (as sample indices, not seconds). This is used to improve raster plot performance.
        self._cache = {} if cache is None else cache

    def __repr__(self):
        return f"Wrapped {repr(self.si_obj)}"

    @property
    def sample2time(self) -> Callable:
        """Takes an array of spikeinterface sample numbers and maps them to timestamps. Ideally a vectorized function.
        If not provided, times will be according to the probe's sample clock (e.g. sample / fs).
        """
        if self._sample2time is None:
            fs = self.si_obj.get_sampling_frequency()
            return lambda x: x / fs
        else:
            return self._sample2time

    @property
    def properties(self) -> pd.DataFrame:
        """Return SI cluster properties as a DataFrame, with one row per cluster."""
        df = pd.DataFrame(self.si_obj._properties)
        df["cluster_id"] = self.si_obj.get_unit_ids()
        return df

    @property
    def structures_by_depth(self, ascending=False) -> list[str]:
        """Array of structure acronyms, by descending depth.
        Only structures with at least 1 cluster will be returned."""
        structure_depths = self.properties.groupby("acronym")["depth"].min()
        return structure_depths.sort_values(ascending=ascending).index.values

    @property
    def has_sample2time(self) -> bool:
        return self._sample2time is not None

    def get_unit_ids(self) -> dtypes.ClusterIDs:
        """This exists to match the MultiSIKS API, not the SpikeInterface API."""
        return self.si_obj.get_unit_ids()

    def refine_clusters(
        self,
        simple_filters: Optional[dict] = None,
        callable_filters: Optional[list[Callable]] = None,
        include_nans: bool = True,
    ):
        """Refine clusters, and conveniently wrap the result, so that the user doesn't have to."""
        new_obj = siutils.refine_clusters(
            self.si_obj, simple_filters, callable_filters, include_nans
        )
        return self.__class__(
            new_obj,
            self.sample2time,
            cache=self._cache,
        )

    def select_clusters(self, clusterIDs: dtypes.ClusterIDs):
        """Select clusters, and conveniently wrap the result, so that the user doesn't have to."""
        new_obj = self.si_obj.select_units(clusterIDs)
        return self.__class__(
            new_obj,
            self.sample2time,
            cache=self._cache,
        )

    def select_structures(self, tgt_structure_acronyms: list[str]):
        """Select clusters belonging to desired structures, and update `self.struct` accordingly."""
        all_structure_acronyms = [s for s in self.si_obj.get_annotation("structure_table")["acronym"].unique()]
        if tgt_structure_acronyms is None:
            tgt_structure_acronyms = all_structure_acronyms
        new_obj = siutils.refine_clusters(
            self.si_obj, {"acronym": set(tgt_structure_acronyms)}, include_nans=False
        )
        return self.__class__(
            new_obj,
            self.sample2time,
            cache=self._cache,
        )

    def get_unit_spike_train(
        self,
        cluster_id: int,
        start_frame: Union[int, None] = None,
        end_frame: Union[int, None] = None,
        return_times: bool = False,
        # Whereas the arguments above shadow the SpikeInterface API, the following are added functionality
        # The reason we do not have one "start" kwarg and one "end" kwarg that accepts either samples or seconds, is so that
        # the SI behavior of get_unit_spike_train(id, start_frame, end_frame, return_times=True) can be preserved, which would be ambiguous otherwise.
        start_time: Union[float, None] = None,
        end_time: Union[float, None] = None,
    ) -> dtypes.SpikeTrain:
        """Thin wrapper around SpikeInterface's get_unit_spike_train(), to add caching behavior.
        Beware: Even if only a small time range of data are requested, the whole-recording spike train will be loaded and cached, before returning the data of interest.
        """
        # First, check the cache, and update it if needed.
        if not (cluster_id in self._cache):
            self._cache[cluster_id] = self.si_obj.get_unit_spike_train(
                cluster_id, return_times=False
            )  # Always cache samples, never times.

        # Now, fetch the data from the cache
        train = self._cache[cluster_id]
        l = (
            None
            if start_frame is None
            else np.searchsorted(train, start_frame, side="left")
        )
        r = (
            None
            if end_frame is None
            else np.searchsorted(train, end_frame, side="right")
        )
        train = train[l:r]

        # Convert to seconds, if requested
        if return_times:
            train = self.sample2time(train)

            # Fix times, so that they are monotonically increasing without duplicates
            # TODO: Fix these upstream, before the extractor is created and saved.
            ecephys.utils.hotfix_times(train)  # Ensure monotonicity, fast, inplace
            train = np.unique(train)  # Drop duplicates, slow but not prohibitively so.

            # Get requested data
            if ((start_time is not None) or (end_time is not None)) and (
                (start_frame is not None) or (end_frame is not None)
            ):
                warnings.warn(
                    "You are trying to select spikes based on both seconds and sample indices. Are you sure you want to do this?"
                )
            l = (
                None
                if start_time is None
                else np.searchsorted(train, start_time, side="left")
            )
            r = (
                None
                if end_time is None
                else np.searchsorted(train, end_time, side="right")
            )
            train = train[l:r]

        return train

    def get_property_spike_train(
        self, property_name: str, property_value: Any, **kwargs
    ) -> dtypes.SpikeTrain:
        mask = self.properties[property_name] == property_value
        clusters = self.properties[mask]["cluster_id"].values
        if len(clusters) == 0:
            if ("return_times" in kwargs) and kwargs["return_times"]:
                return np.asarray([], dtype=np.float64)
            else:
                return np.asarray([], dtype=np.int64)
        return ecephys.utils.kway_mergesort(
            [self.get_unit_spike_train(id, **kwargs) for id in clusters]
        )

    def get_cluster_trains(
        self,
        cluster_ids: Optional[dtypes.ClusterIDs] = None,
        display_progress: bool = True,
        **kwargs,
    ) -> dtypes.ClusterTrains:
        """Get spike trains for a list of clusters (default all).

        Performance note: Getting several spike trains this way (i.e. cluster-by-cluster) is MUCH faster than getting them all together using si_obj.get_all_unit_spike_trains()
        """
        if cluster_ids is None:
            cluster_ids = self.si_obj.get_unit_ids()

        if display_progress:
            cluster_ids = tqdm(cluster_ids, desc="Loading trains by cluster_id: ")

        return {id: self.get_unit_spike_train(id, **kwargs) for id in cluster_ids}

    def get_trains_by_property(
        self,
        property_name: str = "depth",
        values: Optional[
            npt.ArrayLike
        ] = None,  # Filter trains, keeping only those with the indicated property values.
        display_progress=True,
        **kwargs,
    ) -> dtypes.SpikeTrainDict:
        """Get all spike trains, merged and keyed by the desired property (e.g. depth, or structure)."""
        if property_name == "cluster_id":
            return self.get_cluster_trains(
                cluster_ids=values, display_progress=display_progress, **kwargs
            )

        if values is None:
            values = self.properties[property_name].unique()

        if display_progress:
            values = tqdm(values, desc=f"Loading trains by `{property_name}`: ")

        return {
            val: self.get_property_spike_train(property_name, val, **kwargs)
            for val in values
        }

    def get_spike_vector(
        self, return_times=False
    ) -> tuple[dtypes.SpikeTrain, dtypes.ClusterIXs, dtypes.ClusterIDs]:
        """This is a replacement for si_obj.get_all_spike_trains()! Never use si_obj.get_all_spike_trains()!
        Not only is this faster (Faster even with an empty cache! MUCH faster with any caching!),
        but it is also more correct, since you need to be very careful about using sample2time on
        the spike times returned by get_all_spike_trains(). Plus, segment-handling logic is easier.
        """
        trains = self.get_cluster_trains(return_times=return_times)
        return cluster_trains.convert_cluster_trains_to_spike_vector(trains)

    # TODO: Move elsewhere, since this is not a wrapper around SI core functionality
    def run_off_detection(
        self,
        hg: hypnogram.FloatHypnogram,
        tgt_states=None,
        split_by_state=True,
        on_off_method="hmmem",
        on_off_params=None,
        spatial_detection=False,
        spatial_params=None,
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
        if not spatial_detection:
            assert (
                spatial_params is None
            ), f"Set `spatial_params=None` if `spatial_detection` is False."

        print(
            f"Running ON/OFF detection. Cutting/concatenating the following hypnogram states: {tgt_states}"
        )

        properties = self.properties
        all_trains = self.get_cluster_trains(return_times=True)

        if not len(all_trains):
            print(
                f"N=0 clusters in sorting (structures = {self.structures_by_depth}). Passing."
            )
            return pd.DataFrame()

        # Get requested subset of epochs and check they have actual spiking activity
        # Remove "NoData"
        all_allowed_states = [s for s in hg.state.unique() if s != "NoData"]
        if tgt_states is None:
            tgt_states = all_allowed_states
        assert all(
            [s in all_allowed_states for s in tgt_states]
        ), f"Invalid value in `tgt_states={tgt_states}`. Available states: {all_allowed_states}"
        mask = hg.state.isin(tgt_states)
        # Remove epochs starting/ending before/after start/end of recording (avoid spurious OFF)
        first_spike_t = min([min(t) for t in all_trains.values()])
        last_spike_t = max([max(t) for t in all_trains.values()])
        mask = (
            mask
            & (hg["start_time"] >= first_spike_t)
            & (hg["end_time"] <= last_spike_t)
        )
        hg = hg[mask]

        # Iterate on states of interest
        if split_by_state:
            states_to_aggregate = [[s] for s in hg.state.unique()]
        else:
            states_to_aggregate = [hg.state.unique()]

        all_structures_dfs = []
        for states in states_to_aggregate:
            trains = [all_trains[row.cluster_id] for row in properties.itertuples()]
            cluster_ids = [row.cluster_id for row in properties.itertuples()]
            cluster_firing_rates = [row.fr for row in properties.itertuples()]
            depths = [row.depth for row in properties.itertuples()]

            sumFR = properties["fr"].sum()
            if sumFR <= min_sum_fr:
                print(
                    f"Too few spikes (sumFR={sumFR}Hz) in the following structures: {self.structures_by_depth}. Passing ON/OFF detection"
                )
                continue
            else:
                print(
                    f"Running ON/OFF detection for structures {self.structures_by_depth}, states {states}.\n"
                    f"N={len(trains)}units, sumFR={sumFR}Hz"
                )

            try:
                if not spatial_detection:
                    df = on_off_detection.OnOffModel(
                        trains,
                        None,
                        cluster_ids=cluster_ids,
                        method=on_off_method,
                        params=on_off_params,
                        bouts_df=hg[hg["state"].isin(states)],
                    ).run()
                else:
                    df = on_off_detection.SpatialOffModel(
                        trains,
                        depths,
                        None,
                        cluster_ids=cluster_ids,
                        cluster_firing_rates=cluster_firing_rates,
                        on_off_method=on_off_method,
                        on_off_params=on_off_params,
                        spatial_params=spatial_params,
                        bouts_df=hg[hg["state"].isin(states)],
                        n_jobs=n_jobs,
                    ).run()
            except on_off_detection.ALL_METHOD_EXCEPTIONS as e:
                print(
                    f"\n\nException for structures {self.structures_by_depth}: {e}\n\n Passing.\n"
                )
                continue
            df["states"] = [",".join(states)] * len(df)

            if len(df):
                all_structures_dfs.append(df[df["state"] == "off"])

        if not len(all_structures_dfs):
            return pd.DataFrame()

        return pd.concat(all_structures_dfs).reset_index(drop=True)

    # TODO: Make a staticmethod, and take properties dataframe directly as first argument
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
            "Amplitude",
            "ContamPct",
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

    # TODO: Make a staticmethod, and take properties dataframe directly as first argument
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

    # TODO: Make a staticmethod, and take properties dataframe directly as first argument
    def plot_confusion(self, ax: Optional[plt.Axes] = None) -> plt.Axes:
        """Plot a confusion matrix to compare manual curation labels with automatically assigned KS labels."""
        if ax is None:
            fig, ax = plt.subplots()

        df = self.properties
        skmetrics.ConfusionMatrixDisplay.from_predictions(
            df["quality"], df["KSLabel"], ax=ax
        )
        return ax
