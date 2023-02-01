import ecephys as ece
import logging
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import probeinterface as pi
import yaml
import json
from horology import Timing

import spikeinterface.extractors as se
import spikeinterface.full as si
import spikeinterface.postprocessing as sp
import spikeinterface.qualitymetrics as sq
import spikeinterface.sorters as ss
from ecephys import wne
from ecephys.utils import siutils as si_utils

from .preprocess_si_rec import preprocess_si_recording

logger = logging.getLogger(__name__)

Pathlike = Union[Path, str]
OPTS_FNAME = "opts.yaml"


"""
- <project>
    - <experiment>
        - <alias>
            - <subject>
            - <basename>.<probe>/ # This is the main output directory
                - sorter_output/ # This is the sorter output directory
            - prepro_<basename>.<probe>/ # This is the preprocessing output directory
"""


class AbstractSortingPipeline:
    def __init__(
        self,
        wneProject,
        wneSubject,
        experiment: str,
        alias: str,
        probe: str,
        basename: str,
        rerun_existing: bool = True,
        n_jobs: int = 1,
        options_source: Union[str, Path] = "wneProject",
        exclusions: Optional[pd.DataFrame] = None,
    ):
        # These properties are private, because they should not be modified after instantiation
        self._wneProject = wneProject
        self._wneSubject = wneSubject
        self._alias = alias
        self._experiment = experiment
        self._probe = probe
        self._rerun_existing = rerun_existing
        self._nJobs = n_jobs
        self._basename = basename

        # Set the location where options should be loaded from. Either...
        #   (1) wneProject
        #   (2) A custom location of your choosing
        if options_source == "wneProject":
            self._opts_src = wneProject.get_experiment_subject_file(
                experiment, wneSubject.name, wne.constants.SORTING_PIPELINE_PARAMS_FNAME
            )
        else:
            self._opts_src = Path(options_source)

        # Load the options, either using JSON or YAML (for backwards compatibility)
        with open(self._opts_src, "r") as f:
            suffix = self._opts_src.suffix
            if suffix == ".json":
                self._opts = json.load(f)
            elif suffix in {".yaml", ".yml"}:
                self._opts = yaml.load(f)
            else:
                raise ValueError(
                    f"Unexpected file extension {suffix} on {self._opts_src}"
                )

        # Exclude artifacts found in the WNE project, unless overriden
        self._exclusions = (
            self.get_wne_artifacts() if exclusions is None else exclusions
        )

    def __repr__(self):
        return f"""
        {self.__class__.__name__}:
        {self._wneSubject.name}, {self._probe}
        {self._experiment} - {self._alias}
        N_jobs: {self._nJobs}
        Sorting output_dir: {self.sorter_output_dir}
        Preprocessing output_dir: {self.preprocessing_output_dir}
        First segment full path: \n{self.raw_ap_bin_segment_frame.path.values[0]}
        Exclusions: {self._exclusions}
        Total sorting duration: \n{self.raw_si_recording.get_total_duration()}(s)
        AP segment table: \n{self.raw_ap_bin_segment_frame.loc[:,['run', 'gate', 'trigger', 'probe', 'wneSegmentStartTime', 'segment_idx', 'segmentTimeSecs', 'segmentFileSizeRatio']]}
        """

    def get_wne_artifacts(self) -> pd.DataFrame:
        artifacts_file = self._wneProject.get_experiment_subject_file(
            self._experiment,
            self._wneSubject.name,
            f"{self._probe}.ap.{ece.wne.constants.ARTIFACTS_FNAME}",
        )
        return ece.utils.read_htsv(artifacts_file)

    @property
    def opts(self) -> dict:
        return (
            self._opts.copy()
        )  # Return a copy, so we can safely modify (e.g. pop) the working copy.

    @property
    def main_output_dir(self) -> Path:
        return (
            self._wneProject.get_alias_subject_directory(
                self._experiment, self._alias, self._wneSubject.name
            )
            / f"{self._basename}.{self._probe}"
        )

    @property
    def sorter_output_dir(self) -> Path:
        return self.main_output_dir / "sorter_output"

    @property
    def preprocessing_output_dir(self) -> Path:
        return (
            self._wneProject.get_alias_subject_directory(
                self._experiment, self._alias, self._wneSubject.name
            )
            / f"prepro_{self._basename}.{self._probe}"
        )

    def write_exclusions_frame(self):
        ece.utils.write_htsv(self._exclusions, self.main_output_dir / "exclusions.htsv")

    ##### Not implemented #####

    @property
    def is_sorted(self):
        raise NotImplementedError()

    @property
    def is_postprocessed(self):
        raise NotImplementedError()

    def to_pickle():
        raise NotImplementedError()

    def switch_project(newProjectName):
        raise NotImplementedError()

    def run_pipeline():
        raise NotImplementedError()

    def run_metrics():
        raise NotImplementedError()


# TODO: Why is there a separate abstract base class, instead of just this class? The ABC has SI specific methods and properties.
# TODO: Shouldn't there be an is_preprocessed method?
class SpikeInterfaceSortingPipeline(AbstractSortingPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Input, output files and folders, specific to SI
        # All as properties in case we change project for SortingPipeline object
        # And because all preprocessing is done in a lazy way anyways
        self._raw_si_recording = None
        self._segments
        self._preproccessed_si_recording = None
        self._dumped_bin_si_recording = None
        self._si_sorting_extractor = None
        self._si_waveform_extractor = None

        # TODO use only properties
        # Pipeline steps, specific to SI
        self._is_preprocessed = False
        self._is_dumped = False
        self._is_curated = False
        self._is_run_metrics = False

        self.job_kwargs = {
            "n_jobs": self._nJobs,
            "chunk_duration": "1s",
            "progress_bar": True,
        }

    ##### Preprocessing methods and properties #####

    def get_raw_si_recording(self):
        self._raw_si_recording, self._segments = self._wneSubject.get_si_recording(
            self._experiment,
            self._alias,
            "ap",
            self._probe,
            combine="concatenate",
            exclusions=self._exclusions,
            sampling_frequency_max_diff=1e-06,
        )
        return self._raw_si_recording, self._segments

    @property
    def preprocessed_si_probe_path(self):
        return self.main_output_dir / "preprocessed_si_probe.json"

    def dump_preprocessed_si_probe(self):
        # Used to load probe when running waveform extractor/quality metrics
        # NB: The probe used for sorting may differ from the raw SGLX probe,
        # Since somme channels may be dropped during motion correction
        # self.processed_si_probe_path.parent.mkdir(exist_ok=True, parents=True)
        pi.write_probeinterface(
            self.preprocessed_si_probe_path,
            self._preprocessed_si_recording.get_probe(),
        )
        print(f"Save processed SI probe at {self.preprocessed_si_probe_path}")

    def run_preprocessing(self):
        raw_si_recording, segments = self.get_raw_si_recording()
        self._preproccessed_si_recording = preprocess_si_recording(
            raw_si_recording,
            self.opts,
            output_dir=self.preprocessing_output_dir,
            rerun_existing=self._rerun_existing,
            job_kwargs=self.job_kwargs,
        )
        self._is_preprocessed = True

        # Save options used
        with open(self.preprocessing_output_dir / OPTS_FNAME, "w") as f:
            yaml.dump(self.opts, f)
        self.dump_preprocessed_si_probe()
        ece.utils.write_htsv(self._exclusions, self.main_output_dir / "exclusions.htsv")
        ece.utils.write_htsv(segments, self.main_output_dir / "segments.htsv")

    ##### Sorting methods and properties #####

    @property
    def is_sorted(self):
        required_output_files = [
            self.sorter_output_dir / "spike_times.npy",
            self.sorter_output_dir / "amplitudes.npy",
            self.main_output_dir / "sorting_pipeline_opts.yaml",
            self.main_output_dir / "processed_si_probe.json",
        ]
        return self.sorter_output_dir.exists() and all(
            [f.exists() for f in required_output_files]
        )

    def run_sorting(self):
        # If sorting is already complete and we are rerun_existing=False, just return.
        if self.is_sorted and not self._rerun_existing:
            logger.info(
                f"Data are already sorted and rerun_existing=False. Doing nothing.\n\n"
            )
            return

        # Get sorter and parameters
        sorter_name, sorter_params = self._opts["sorting"]

        # If using KiloSort2.5, we need to set the path to the KiloSort executable
        if sorter_name == "kilosort2_5":
            ks_path = sorter_params.pop("ks_path")  # TODO: Why pop?
            ss.sorter_dict[sorter_name].set_kilosort2_5_path(ks_path)

        # Sort
        self.main_output_dir.mkdir(exist_ok=True, parents=True)  # Necessary?
        with Timing(name="Run spikeinterface sorter"):
            ss.run_sorter(
                sorter_name,
                self._preprocessed_si_recording,
                output_folder=self.main_output_dir,
                verbose=True,
                with_output=False,
                **sorter_params,
                **self.job_kwargs,
            )

        # Save options used
        with open(self.main_output_dir / OPTS_FNAME, "w") as f:
            yaml.dump(self.opts, f)

    ##### Postprocessing methods and properties #####
    # Metrics are just one componenent of postprocessing.

    def run_metrics(self):
        # Get list of metrics
        # get set of params across metrics
        metrics_opts = self._opts["metrics"].copy()
        metrics_names = list(metrics_opts.keys())
        params = {}
        for metric_dict in metrics_opts.values():
            params.update(metric_dict)

        print(f"Running metrics: {metrics_names}")
        print(f"Metrics params: {params}")

        print("Computing metrics")
        with Timing(name="Compute metrics: "):
            metrics_df = sq.compute_quality_metrics(
                self.si_waveform_extractor,
                metric_names=metrics_names,
                n_jobs=self._nJobs,
                progress_bar=True,
                verbose=True,
                **params,
            )

        print("Save metrics dataframe as `metrics.csv` in kilosort dir")
        metrics_df.to_csv(
            self.sorter_output_dir / "metrics.csv",
            # sep="\t",
            index=True,
            index_label="cluster_id",
        )

        self.dump_opts(self.main_output_dir, fname="metrics_opts.yaml")
        print("Done")

    def run_postprocessing(self):
        UNIT_LOCATIONS_METHOD = "center_of_mass"

        with Timing(name="Compute principal components: "):
            print("Computing principal components.")
            sp.compute_principal_components(
                self.si_waveform_extractor,
                load_if_exists=True,
                n_components=5,
                mode="by_channel_local",
                n_jobs=self._nJobs,
            )

        with Timing(name="Compute unit locations: "):
            print(f"Computing unit locations (method={UNIT_LOCATIONS_METHOD}")
            sp.compute_unit_locations(
                self.si_waveform_extractor,
                load_if_exists=True,
                method=UNIT_LOCATIONS_METHOD,
            )

        with Timing(name="Compute template metrics: "):
            print("Computing template metrics")
            sp.compute_template_metrics(
                self.si_waveform_extractor,
                load_if_exists=True,
            )

        self.run_metrics()

    ##### Misc #####

    @property
    def waveforms_dir(self):
        return self.main_output_dir / "waveforms"

    @property
    def is_postprocessed(self):
        return (
            self.is_sorted
            and self.waveforms_dir.exists()
            and (self.sorter_output_dir / "metrics.csv").exists()
        )

    @property
    def preprocessed_bin_path(self):
        return self.sorter_output_dir / "recording.dat"

    @property
    def dumped_bin_si_recording(self):
        if self._dumped_bin_si_recording is None:
            self._dumped_bin_si_recording = self.load_dumped_bin_si_recording()
        return self._dumped_bin_si_recording

    @property
    def si_sorting_extractor(self):
        if self._si_sorting_extractor is None:
            self._si_sorting_extractor = self.load_si_sorting_extractor()
        return self._si_sorting_extractor

    @property
    def si_waveform_extractor(self):
        if self._si_waveform_extractor is None:
            self._si_waveform_extractor = self.load_si_waveform_extractor()
        return self._si_waveform_extractor

    ######### IO ##########

    def load_si_sorting_extractor(self):
        return se.read_kilosort(self.sorter_output_dir)

    def load_dumped_bin_si_recording(self):
        return si_utils.load_kilosort_bin_as_si_recording(
            self.sorter_output_dir,
            fname=self.preprocessed_bin_path.name,
            si_probe=self.load_si_probe(),
        )

    def load_si_waveform_extractor(self):
        MS_BEFORE = 1.5
        MS_AFTER = 2.5
        MAX_SPIKES_PER_UNIT = 2000
        SPARSITY_RADIUS = 400
        NUM_SPIKES_FOR_SPARSITY = 500

        assert self.is_sorted
        load_precomputed = not self._rerun_existing and self.is_postprocessed
        if self.preprocessed_bin_path.exists():
            waveform_recording = self.dumped_bin_si_recording
        else:
            # Make sure we preprocess in the same way as during sorting
            self.run_preprocessing(load_existing=True)
            waveform_recording = self._preprocessed_si_recording

        if load_precomputed:
            print(
                f"Loading waveforms from folder. Will use this recording: {waveform_recording}"
            )
            we = si.WaveformExtractor.load_from_folder(
                self.waveforms_dir,
                with_recording=False,
                sorting=self.si_sorting_extractor,
            )
            # Hack to have recording even if we deleted or moved the data,
            # because load_from_folder(.., with_recording=True) assumes same recording file locations etc
            we._recording = waveform_recording
        else:
            with Timing(name="Extract waveforms: "):
                we = si.extract_waveforms(
                    waveform_recording,
                    self.si_sorting_extractor,
                    folder=self.waveforms_dir,
                    overwrite=True,
                    ms_before=MS_BEFORE,
                    ms_after=MS_AFTER,
                    max_spikes_per_unit=MAX_SPIKES_PER_UNIT,
                    sparse=True,
                    num_spikes_for_sparsity=NUM_SPIKES_FOR_SPARSITY,
                    method="radius",
                    radius_um=SPARSITY_RADIUS,
                    **self.job_kwargs,
                )
        return we

    def load_si_probe(self):
        # Used to load probe when running waveform extractor/quality metrics
        # NB: The probe used for sorting may differ from the raw SGLX probe,
        # Since somme channels may be dropped during motion correction
        if not self.preprocessed_si_probe_path.exists():
            import warnings

            warnings.warn(
                # raise FileNotFoundError(
                f"Find no spikeinterface probe object at {self.preprocessed_si_probe_path}"
                f"\nRe-dumping probe"
            )
            self.dump_preprocessed_si_probe()
        prb_grp = pi.read_probeinterface(self.preprocessed_si_probe_path)
        assert len(prb_grp.probes) == 1
        prb = prb_grp.probes[0]
        print(prb)
        return prb

    ######### Pipeline steps ##########

    def run_drift_estimates(self):
        self.run_preprocessing()
        self.dump_opts()

    def run_pipeline(self):
        self.run_preprocessing()
        self.run_sorting()


class SortingPipeline(SpikeInterfaceSortingPipeline):
    def __init__(*args, **kwargs):
        super().__init__(*args, **kwargs)


def load_from_pickle(path):
    raise NotImplementedError
