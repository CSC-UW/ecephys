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
from .... import hypnogram as hp

# TODO: Be consistent about using logger vs print
logger = logging.getLogger(__name__)

Pathlike = Union[Path, str]
POSTPRO_OPTS_FNAME = "postpro_opts.yaml"
HYPNO_FNAME = "simplified_hypnogram.htsv"


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
                        - si_output/ # This is Spikeinterface's output ("waveforms") directory. Everything in there may be deleted by spikeinterface
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
        hypnogram_source: Union[str, Path, Project] = None,
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

        # Pull hypnogram if provided
        if hypnogram_source is None:
            self.hypnogram = None
        else:
            if isinstance(hypnogram_source, Project):
                hypno_path = hypnogram_source.get_experiment_subject_file(
                    experiment,
                    wneSubject.name,
                    HYPNO_FNAME,
                )
            elif isinstance(hypnogram_source, (str, Path)):
                hypno_path = Path(hypnogram_source)
            else:
                raise ValueError(f"Unrecognize type for `hypnogram_source`: {hypnogram_source}")
            self.hypnogram = self.load_and_format_hypnogram(hypno_path)

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
    
    def __repr__(self):
        repr = f"""
        {self.__class__}
        Postprocessing output dir: {self.postprocessing_output_dir}
        Options source: {self._opts_src}
        """
        if self.hypnogram is not None:
            repr += f"""Hypnogram: {self.hypnogram.groupby('state').describe()["duration"]}"""
        else:
            repr += f"Hypnogram: None"
        return repr

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
    def waveforms_output_dir(self) -> Path:
        """Below postprocessing_output_dir because content deleted by spikeinterface"""
        return self.postprocessing_output_dir / "si_output"
    
    # TODO: All this def belongs elsewhere... but I don't know where
    def load_and_format_hypnogram(self, hypno_path):
        """Load, convert to si sorting sample indices, select epochs."""
        if not hypno_path.exists():
            raise FileNotFoundError(
                f"""Expected to find a hypnogram at `{hypno_path}`."""
            )
        hypno = ece_utils.read_htsv(hypno_path)

        # Trim hypnogram so that each bout falls within segments
        segments = self._sorting_pipeline._segments
        segments = segments[segments["type"] == "keep"]
        segment_starts = segments.expmtPrbAcqFirstTime
        segment_ends = segments.expmtPrbAcqFirstTime + segments.segmentDuration
        hypno = hp.trim_multiple_epochs(
            hypno,
            segment_starts,
            segment_ends
        )
        
        time2sample = self._wneProject.get_time2sample(
            self._wneSubject,
            self._experiment,
            self._alias,
            self._probe,
            self._sorting_basename,
            allow_no_sync_file=True
        )

        hypno["start_frame"] = time2sample(hypno["start_time"])
        hypno["end_frame"] = time2sample(hypno["end_time"])

        # Since we trimmed the hypnogram to match segments, 
        # either both or None of start_frame/end_frame should be Nan
        assert all(hypno.isnull().all(axis=1) == hypno.isnull().all(axis=1))

        return hypno[~hypno.isnull().any(axis=1)]

    @property
    def is_postprocessed(self) -> bool:
        return (
            self._sorting_pipeline.is_sorted
            and self.waveforms_output_dir.exists()
            and (
                self.waveforms_output_dir / "quality_metrics" / "metrics.csv"
            ).exists()
        )

    def load_waveform_extractor(self) -> si.WaveformExtractor:
        """Load a waveform extractor. This may `extract` previously computed and saved waveforms."""

        if not "waveforms" in self._opts:
            raise ValueError("Expected 'waveforms' entry in postprocessing option file.")
        waveforms_kwargs = self._opts["waveforms"]

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
                self.waveforms_output_dir,
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
                    folder=self.waveforms_output_dir,
                    overwrite=True,
                    **waveforms_kwargs,
                    **self.job_kwargs,
                )
        return we

    @property
    def waveform_extractor(self) -> si.WaveformExtractor:
        if self._si_waveform_extractor is None:
            self._si_waveform_extractor = self.load_waveform_extractor()
        return self._si_waveform_extractor

    def run_metrics(self):

        if not "metrics" in self._opts:
            raise ValueError("Expected 'metrics' entry in postprocessing option file.")
        metrics_opts = self._opts["metrics"]

        # Set job kwargs for all
        si.set_global_job_kwargs(n_jobs=self._nJobs)

        metrics_names = list(metrics_opts.keys())
        print(f"Running metrics: {metrics_names}")
        print(f"Metrics params: {metrics_opts}")

        print("Computing metrics")
        with Timing(name="Compute metrics: "):
            sq.compute_quality_metrics(
                self.waveform_extractor,
                metric_names=metrics_names,
                qm_params=metrics_opts,
                progress_bar=True,
                verbose=True,
            )

    def run_postprocessing(self):

        if not "postprocessing" in self._opts:
            raise ValueError("Expected 'postprocessing' entry in postprocessing option file.")

        # Set job kwargs for all
        si.set_global_job_kwargs(n_jobs=self._nJobs)

        # Iterate on spikeinterface.postprocessing functions by name
        for func_name, func_kwargs in self._opts["postprocessing"].items():

            if not hasattr(sp, func_name):
                raise ValueError(
                    "Could not find function `{func_name}` in spikeinterface.postprocessing module"
                )
            func = getattr(sp, func_name)

            with Timing(name=f"{func_name}: "):
                func(
                    self.waveform_extractor,
                    load_if_exists=True,
                    **func_kwargs,
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
