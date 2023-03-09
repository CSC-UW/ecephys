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

from .sorting_pipeline import SpikeInterfaceSortingPipeline, get_main_output_dir
from ..subjects import Subject
from ...projects import Project
from ... import constants
from .... import utils as ece_utils

# TODO: Be consistent about using logger vs print
logger = logging.getLogger(__name__)

Pathlike = Union[Path, str]
POSTPRO_OPTS_FNAME = "postpro_opts.yaml"


"""
- <project>
    - <experiment>
        - <alias>
            - <subject>
                - <sorting_basename>.<probe>/ # This is the main output directory
                    - si_output/ # This is Spikeinterface's output directory. Everything in there is deleted by spikeinterface when running run_sorter.
                        - sorter_output/ # This is the sorter output directory
                    - preprocessing # This is the preprocessing output directory
                    - <basename> # This is the postprocessing output directory
"""


class SpikeInterfacePostprocessingPipeline:
    def __init__(
        self,
        wneProject: Project,
        wneSubject: Subject,
        experiment: str,
        alias: str,
        probe: str,
        sorting_basename: str,
        postprocessing_name: str = "postprocessing_df",
        rerun_existing: bool = True,
        n_jobs: int = 1,
        options_source: Union[str, Path] = "wneProject",
    ):
        # These properties are private, because they should not be modified after instantiation
        self._wneProject = wneProject
        self._wneSubject = wneSubject
        self._alias = alias
        self._experiment = experiment
        self._probe = probe
        self._postprocessing_name = postprocessing_name
        self._sorting_basename = sorting_basename
        self._rerun_existing = rerun_existing
        self._nJobs = int(n_jobs)

        self._sorting_pipeline = SpikeInterfaceSortingPipeline.load_from_folder(
            self._wneProject,
            self._wneSubject,
            self._experiment,
            self._alias,
            self._probe,
            self._sorting_basename,
        )

        # Set the location where options should be loaded from. Either...
        #   (1) wneProject
        #   (2) A custom location of your choosing
        if options_source == "wneProject":
            self._opts_src = wneProject.get_experiment_subject_file(
                experiment,
                wneSubject.name,
                constants.PREPROCESSING_PIPELINE_PARAMS_FNAME,
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
        # Ensure we use same opts as previously
        prior_opts_path = self.postprocessing_output_dir / POSTPRO_OPTS_FNAME
        if prior_opts_path.exists():
            with open(prior_opts_path, "r") as f:
                prior_opts = yaml.load(f, Loader=yaml.SafeLoader)
            if deepdiff.DeepDiff(self._opts, prior_opts):
                raise ValueError(
                    f"Current options do not match prior options saved on file at {prior_opts_path}.\n"
                    f"Current opts: {self._opts}\n"
                    f"Prior opts: {prior_opts}\n"
                )

        # Set compute details
        self.job_kwargs = {
            "n_jobs": self._nJobs,
            "chunk_duration": "1s",
            "progress_bar": True,
        }

        # Input, output files and folders, specific to SI
        self._si_waveform_extractor = None

        self.check_sorting_output()
    
    def check_sorting_output(self):
        assert self._sorting_pipeline.is_sorted
        # TODO: This shouldn't be necessary... 
        # But I noticed that before opening/saving with phy, the returned sorter may have clusters
        # with N=0 spikes, which messes up postprocessing. Those are filtered out somehow when running phy.
        assert (self._sorting_pipeline.sorter_output_dir / "cluster_info.tsv").exists(), (
            f"You need to open/save this sorting with Phy first."
        )

    # Paths

    @property
    def main_output_dir(self) -> Path:
        return self._sorting_pipeline.main_output_dir

    @property
    def sorter_output_dir(self) -> Path:
        return self._sorting_pipeline.sorter_output_dir

    @property
    def postprocessing_output_dir(self) -> Path:
        return self.main_output_dir / self._postprocessing_name

    @property
    def is_postprocessed(self) -> bool:
        return (
            self._sorting_pipeline.is_sorted
            and self.postprocessing_output_dir.exists()
            and (
                self.postprocessing_output_dir / "quality_metrics" / "metrics.csv"
            ).exists()
        )

    def load_waveform_extractor(self) -> si.WaveformExtractor:
        """Load a waveform extractor. This may `extract` previously computed and saved waveforms."""
        MS_BEFORE = 1.5
        MS_AFTER = 2.5
        MAX_SPIKES_PER_UNIT = 2000
        SPARSITY_RADIUS = 400
        NUM_SPIKES_FOR_SPARSITY = 500

        assert (
            self._sorting_pipeline.is_sorted
        ), "Cannot load waveform extractor for unsorted recording."

        waveform_recording = self._sorting_pipeline.processed_extractor_for_waveforms
        sorting = self._sorting_pipeline.sorting_extractor

        # If we have already extracted waveforms before, re-use those.
        precomputed_waveforms = not self._rerun_existing and self.is_postprocessed
        if precomputed_waveforms:
            print(
                f"Loading precomputed waveforms. Will use this recording: {waveform_recording}"
            )
            we = si.WaveformExtractor.load_from_folder(
                self.postprocessing_output_dir,
                with_recording=False,
                sorting=sorting,
            )
            # Hack to have recording even if we deleted or moved the data,
            # because load_from_folder(.., with_recording=True) assumes same recording file locations etc
            we._recording = waveform_recording
        else:
            with Timing(name="Extract waveforms: "):
                we = si.extract_waveforms(
                    waveform_recording,
                    sorting,
                    folder=self.postprocessing_output_dir,
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
            sq.compute_quality_metrics(
                self.waveform_extractor,
                metric_names=metrics_names,
                n_jobs=self._nJobs,
                progress_bar=True,
                verbose=True,
                **params,
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
        with open(self.postprocessing_output_dir / POSTPRO_OPTS_FNAME, "w") as f:
            yaml.dump(self._opts, f)

        print("Done postprocessing.")

    ##### Reinstantiate #####

    @classmethod
    def load_from_folder(
        cls,
        wneProject: Project,
        wneSubject: Subject,
        experiment: str,
        alias: str,
        probe: str,
        sorting_basename: str,
        postprocessing_name: str,
        rerun_existing: bool = False,
    ):
        main_output_dir = get_main_output_dir(
            wneProject,
            wneSubject,
            experiment,
            alias,
            probe,
            sorting_basename,
        )
        postpro_output_dir = main_output_dir / postprocessing_name

        if not all(
            [
                f.exists()
                for f in [
                    postpro_output_dir,
                    postpro_output_dir / POSTPRO_OPTS_FNAME,
                ]
            ]
        ):
            raise FileNotFoundError(
                f"Could not find all required files in {postpro_output_dir}"
            )
        
        return SpikeInterfacePostprocessingPipeline(
            wneProject,
            wneSubject,
            experiment,
            alias,
            probe,
            sorting_basename,
            postprocessing_name,
            rerun_existing=rerun_existing,
            options_source=(postpro_output_dir / POSTPRO_OPTS_FNAME),
        )
