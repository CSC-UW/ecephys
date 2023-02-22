import deepdiff
from horology import Timing
import json
import logging
import pandas as pd
from pathlib import Path
import probeinterface as pi
from typing import Optional, Union
import yaml
import spikeinterface.extractors as se
import spikeinterface.full as si
import spikeinterface.postprocessing as sp
import spikeinterface.qualitymetrics as sq
import spikeinterface.sorters as ss

from . import preprocess_si_rec
from ..subjects import Subject
from ...projects import Project
from ... import constants
from .... import utils as ece_utils

# TODO: Be consistent about using logger vs print
logger = logging.getLogger(__name__)

Pathlike = Union[Path, str]
OPTS_FNAME = "opts.yaml"
POSTPRO_OPTS_FNAME = "metrics_opts.yaml"  # Really, postprocessing options.


"""
- <project>
    - <experiment>
        - <alias>
            - <subject>
            - <basename>.<probe>/ # This is the main output directory
                - sorter_output/ # This is the sorter output directory
            - prepro_<basename>.<probe>/ # This is the preprocessing output directory
"""
# TODO: Shouldn't there be an is_preprocessed method?
class SpikeInterfaceSortingPipeline:
    def __init__(
        self,
        wneProject: Project,
        wneSubject: Subject,
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
                experiment, wneSubject.name, constants.SORTING_PIPELINE_PARAMS_FNAME
            )
        else:
            self._opts_src = Path(options_source)

        # Load the options, either using JSON or YAML (for backwards compatibility)
        with open(self._opts_src, "r") as f:
            suffix = self._opts_src.suffix
            if suffix == ".json":
                self._opts = json.load(f)
            elif suffix in {".yaml", ".yml"}:
                self._opts = yaml.load(f, Loader=yaml.SafeLoader)
            else:
                raise ValueError(
                    f"Unexpected file extension {suffix} on {self._opts_src}"
                )

        # Exclude artifacts found in the WNE project, unless overriden
        self._exclusions = (
            self.get_wne_artifacts() if exclusions is None else exclusions
        )

        # Set compute details
        self.job_kwargs = {
            "n_jobs": self._nJobs,
            "chunk_duration": "1s",
            "progress_bar": True,
        }

        # Input, output files and folders, specific to SI
        # All as properties in case we change project for SortingPipeline object
        # And because all preprocessing is done in a lazy way anyways
        self._raw_si_recording = None
        self._segments = None
        self._preprocessed_si_recording = None  # si.BaseRecording
        self._kilosort_binary_recording_extractor = None
        self._si_sorting_extractor = None
        self._si_waveform_extractor = None

        # TODO use only properties
        # Pipeline steps, specific to SI
        self._is_preprocessed = False
        self._is_dumped = False
        self._is_curated = False
        self._is_run_metrics = False

    def __repr__(self):
        repr = f"""
        {self.__class__.__name__}:
        {self._wneSubject.name}, {self._probe}
        {self._experiment} - {self._alias}
        N_jobs: {self._nJobs}, Rerun Existing: {self._rerun_existing}
        Sorting output_dir: {self.sorter_output_dir}
        Preprocessing output_dir: {self.preprocessing_output_dir}
        Exclusions: {self._exclusions}
        Options source: {self._opts_src}
        """
        if self._raw_si_recording is not None:
            repr = (
                repr
                + f"""
            Raw SI recording: \n{self._raw_si_recording}
            """
            )
        if self._segments is not None:
            repr = (
                repr
                + f"""
            First segment full path: \n{self._segments.path.values[0]}
            AP segment table: \n{self._segments.loc[:,['fname', 'type', 'start_frame', 'end_frame']]}
            """
            )
        if self._raw_si_recording is None or self._segments is None:
            repr = (
                repr
                + f"""
            Run `get_raw_si_recording` method to display segments/recording info.\n`
            """
            )

        return repr

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

    ##### Preprocessing methods and properties #####

    def get_wne_artifacts(self) -> pd.DataFrame:
        artifacts_file = self._wneProject.get_experiment_subject_file(
            self._experiment,
            self._wneSubject.name,
            f"{self._probe}.ap.{constants.ARTIFACTS_FNAME}",
        )
        if artifacts_file.exists():
            return ece_utils.read_htsv(artifacts_file)
        else:
            return pd.DataFrame(
                {"fname": [], "start_time": [], "end_time": [], "type": []}
            )

    def get_raw_si_recording(self) -> tuple[si.BaseRecording, pd.DataFrame]:
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
    def sorter_output_dir(self) -> Path:
        return self.main_output_dir / "sorter_output"

    @property
    def preprocessed_bin_path(self) -> Path:
        return self.sorter_output_dir / "recording.dat"

    @property
    def preprocessed_probe_path(self) -> Path:
        return self.main_output_dir / "preprocessed_si_probe.json"

    @property
    def preprocessed_probe(self) -> pi.Probe:
        """The probe for the preprocessed recording may differ from the probe for the raw recording,
        because somme channels may be dropped during motion correction.
        """
        if self._preprocessed_si_recording is not None:
            self._preprocessed_probe = self._preprocessed_si_recording.get_probe()
        elif self.preprocessed_probe_path.exists():
            probe_group = pi.read_probeinterface(self.preprocessed_probe_path)
            assert len(probe_group.probes) == 1, "Expected to find only one probe"
            self._preprocessed_probe = probe_group.probes[0]
        else:
            raise AttributeError(
                f"No preprocessing object found, and no probefile at {self.preprocessed_probe_path}.\n"
                "You need to run preprocessing."
            )

    @property
    def preprocessing_output_dir(self) -> Path:
        return (
            self._wneProject.get_alias_subject_directory(
                self._experiment, self._alias, self._wneSubject.name
            )
            / f"prepro_{self._basename}.{self._probe}"
        )

    def run_preprocessing(self):
        raw_si_recording, segments = self.get_raw_si_recording()
        self._preprocessed_si_recording = preprocess_si_rec.preprocess_si_recording(
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
        ece_utils.write_htsv(self._exclusions, self.main_output_dir / "exclusions.htsv")
        ece_utils.write_htsv(segments, self.main_output_dir / "segments.htsv")

        # Save preprocessed probe
        pi.write_probeinterface(
            self.preprocessed_probe_path,
            self.preprocessed_probe,
        )

    ##### Sorting methods and properties #####

    @property
    def is_sorted(self) -> bool:
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

    @property
    def waveforms_dir(self) -> Path:
        return self.main_output_dir / "waveforms"

    @property
    def is_postprocessed(self) -> bool:
        return (
            self.is_sorted
            and self.waveforms_dir.exists()
            and (self.sorter_output_dir / "metrics.csv").exists()
        )

    @property
    def kilosort_binary_recording_extractor(self) -> si.BinaryRecordingExtractor:
        if self._kilosort_binary_recording_extractor is None:
            self._kilosort_binary_recording_extractor = (
                ece_utils.siutils.load_kilosort_bin_as_si_recording(
                    self.sorter_output_dir,
                    fname=self.preprocessed_bin_path.name,
                    si_probe=self.preprocessed_probe,
                )
            )
        return self._kilosort_binary_recording_extractor

    @property
    def sorting_extractor(self) -> si.KiloSortSortingExtractor:
        if self._si_sorting_extractor is None:
            self._si_sorting_extractor = se.read_kilosort(self.sorter_output_dir)
        return self._si_sorting_extractor

    def load_waveform_extractor(self) -> si.WaveformExtractor:
        """Load a waveform extractor. This may `extract` previously computed and saved waveforms."""
        MS_BEFORE = 1.5
        MS_AFTER = 2.5
        MAX_SPIKES_PER_UNIT = 2000
        SPARSITY_RADIUS = 400
        NUM_SPIKES_FOR_SPARSITY = 500

        assert self.is_sorted, "Cannot load waveform extractor for unsorted recording."

        # Re-run preprocessing if necessary
        if self.preprocessed_bin_path.exists():  # No need to re-run
            waveform_recording = self.kilosort_binary_recording_extractor
        else:  # Need to re-run.
            # Make sure we preprocess in the same way we did before.
            with open(self.main_output_dir / OPTS_FNAME, "w") as f:
                prior_opts = yaml.load(f)
            if deepdiff.DeepDiff(self.opts, prior_opts):
                raise ValueError(
                    "Current options do not match options used for prior preprocessing. "
                    "Consider instantiating a pipeline object using the options_source parameter."
                )
            self.run_preprocessing()
            waveform_recording = self._preprocessed_si_recording

        # If we have already extracted waveforms before, re-use those.
        precomputed_waveforms = not self._rerun_existing and self.is_postprocessed
        if precomputed_waveforms:
            print(
                f"Loading precomputed waveforms. Will use this recording: {waveform_recording}"
            )
            we = si.WaveformExtractor.load_from_folder(
                self.waveforms_dir,
                with_recording=False,
                sorting=self.sorting_extractor,
            )
            # Hack to have recording even if we deleted or moved the data,
            # because load_from_folder(.., with_recording=True) assumes same recording file locations etc
            we._recording = waveform_recording
        else:
            with Timing(name="Extract waveforms: "):
                we = si.extract_waveforms(
                    waveform_recording,
                    self.sorting_extractor,
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

    @property
    def waveform_extractor(self) -> si.WaveformExtractor:
        if self._si_waveform_extractor is None:
            self._si_waveform_extractor = self.load_waveform_extractor()
        return self._si_waveform_extractor

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
                self.waveform_extractor,
                metric_names=metrics_names,
                n_jobs=self._nJobs,
                progress_bar=True,
                verbose=True,
                **params,
            )

        print("Save metrics dataframe as `metrics.csv` in kilosort dir")
        metrics_df.to_csv(
            self.sorter_output_dir / "metrics.csv",
            index=True,
            index_label="cluster_id",
        )

    def run_postprocessing(self):
        UNIT_LOCATIONS_METHOD = "center_of_mass"

        with Timing(name="Compute principal components: "):
            print("Computing principal components.")
            sp.compute_principal_components(
                self.waveform_extractor,
                load_if_exists=True,
                n_components=5,
                mode="by_channel_local",
                n_jobs=self._nJobs,
            )

        with Timing(name="Compute unit locations: "):
            print(f"Computing unit locations (method={UNIT_LOCATIONS_METHOD}")
            sp.compute_unit_locations(
                self.waveform_extractor,
                load_if_exists=True,
                method=UNIT_LOCATIONS_METHOD,
            )

        with Timing(name="Compute template metrics: "):
            print("Computing template metrics")
            sp.compute_template_metrics(
                self.waveform_extractor,
                load_if_exists=True,
            )

        self.run_metrics()

        # Save options used
        with open(self.main_output_dir / POSTPRO_OPTS_FNAME, "w") as f:
            yaml.dump(self.opts, f)

        print("Done postprocessing.")
