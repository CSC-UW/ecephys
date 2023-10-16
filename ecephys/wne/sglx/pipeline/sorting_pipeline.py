import json
import logging
from pathlib import Path
from typing import Optional, Union

import deepdiff
from horology import Timing
import pandas as pd
from pandas.testing import assert_frame_equal
import probeinterface as pi
import spikeinterface.extractors as se
import spikeinterface.full as si
import spikeinterface.sorters as ss
import yaml

from ecephys import utils
from ecephys.wne import constants
from ecephys.wne import Project
from ecephys.wne.sglx import SGLXSubject
from ecephys.wne.sglx.pipeline import preprocess_si_rec

# TODO: Be consistent about using logger vs print
logger = logging.getLogger(__name__)

Pathlike = Union[Path, str]
OPTS_FNAME = "opts.yaml"
SEGMENTS_FNAME = "segments.htsv"
EXCLUSIONS_FNAME = "exclusions.htsv"


"""
- <project>
    - <experiment>
        - <alias>
            - <subject>
                - <basename>.<probe>/ # This is the main output directory
                    - si_output/ # This is Spikeinterface's output directory. Everything in there is deleted by spikeinterface when running run_sorter.
                        - sorter_output/ # This is the sorter output directory
                    - preprocessing # This is the preprocessing output directory
"""


def get_main_output_dir(
    wneProject: Project,
    wneSubject: SGLXSubject,
    experiment: str,
    alias: str,
    probe: str,
    basename: str,
):
    return (
        wneProject.get_alias_subject_directory(experiment, alias, wneSubject.name)
        / f"{basename}.{probe}"
    )


# TODO: Shouldn't there be an is_preprocessed method?
class SpikeInterfaceSortingPipeline:
    def __init__(
        self,
        wneProject: Project,
        wneSubject: SGLXSubject,
        experiment: str,
        alias: str,
        probe: str,
        basename: str,
        rerun_existing: bool = True,
        n_jobs: int = 1,
        options_source: Union[str, Project, Path] = "wneProject",
        exclusions_source: Union[str, Project, Path] = "wneProject",
    ):
        # These properties are private, because they should not be modified after instantiation
        self._wneProject = wneProject
        self._wneSubject = wneSubject
        self._alias = alias
        self._experiment = experiment
        self._probe = probe
        self._rerun_existing = rerun_existing
        self._nJobs = int(n_jobs)
        self._basename = basename

        # Set the location where options should be loaded from. Either...
        #   (1) "wneProject" -> main wneProject
        #   (1) type "Project" -> another wneProject
        #   (2) A custom location of your choosing
        if options_source == "wneProject":
            self._opts_src = wneProject.get_experiment_subject_file(
                experiment, wneSubject.name, constants.SORTING_PIPELINE_PARAMS_FNAME
            )
        elif isinstance(options_source, Project):
            self._opts_src = options_source.get_experiment_subject_file(
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
        # Ensure we use same opts as previously
        prior_opts_path = self.main_output_dir / OPTS_FNAME
        if prior_opts_path.exists():
            with open(prior_opts_path, "r") as f:
                prior_opts = yaml.load(f, Loader=yaml.SafeLoader)
            if deepdiff.DeepDiff(self._opts, prior_opts):
                raise ValueError(
                    f"Current options do not match prior options saved on file at {prior_opts_path}.\n"
                    f"Consider instantiating pipeline object with `SpikeinterfaceSortingPipeline.load_from_folder`\n\n"
                    f"Current opts: {self.opts}\n"
                    f"Prior opts: {prior_opts}\n"
                )

        # Set the location where exclusions should be loaded from. Either...
        #   (1) "wneProject" -> main wneProject
        #   (1) type "Project" -> another wneProject
        #   (2) A custom location of your choosing
        if exclusions_source == "wneProject":
            self._exclusions = self.get_wne_artifacts(self._wneProject)
        elif isinstance(exclusions_source, Project):
            self._exclusions = self.get_wne_artifacts(exclusions_source)
        else:
            self._exclusions = utils.read_htsv(exclusions_source)
        # Ensure we use the same exclusions as previously
        prior_exclusions_path = self.main_output_dir / EXCLUSIONS_FNAME
        if prior_exclusions_path.exists():
            prior_exclusions = utils.read_htsv(prior_exclusions_path)
            try:
                assert_frame_equal(
                    prior_exclusions,
                    self._exclusions,
                    check_dtype=False,
                    check_index_type=False,
                )
            except AssertionError as e:
                raise ValueError(
                    f"New exclusions don't match previously used exclusion file at {prior_exclusions_path}:\n"
                    f"Consider instantiating pipeline object with `SpikeinterfaceSortingPipeline.load_from_folder`\n"
                    f"{e}"
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
        self._preprocessed_si_recording = None  # si.BaseRecording
        self._kilosort_binary_recording_extractor = None
        self._si_sorting_extractor = None
        self._segments = None
        prior_segments_path = self.main_output_dir / SEGMENTS_FNAME
        if prior_segments_path.exists():
            self._segments = utils.read_htsv(prior_segments_path)

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
        AP segment table: \n{self._segments.loc[:,['fname', 'type', 'start_frame', 'end_frame', 'fileDuration', 'segmentDuration', 'fileDuration']]}
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
        return get_main_output_dir(
            self._wneProject,
            self._wneSubject,
            self._experiment,
            self._alias,
            self._probe,
            self._basename,
        )

    ##### Preprocessing methods and properties #####

    def get_wne_artifacts(self, wne_project) -> pd.DataFrame:
        artifacts_file = wne_project.get_experiment_subject_file(
            self._experiment,
            self._wneSubject.name,
            f"{self._probe}.ap.{constants.ARTIFACTS_FNAME}",
        )
        if artifacts_file.exists():
            return utils.read_htsv(artifacts_file).reset_index(drop=True)
        else:
            raise FileNotFoundError(
                f"Expected artifacts file at {artifacts_file}. "
                f"Please create one or consider specifying a custom path to exclusions  in the `exclusions_source` kwarg."
            )

    def get_raw_si_recording(self) -> tuple[si.BaseRecording, pd.DataFrame]:
        use_cached = isinstance(self._raw_si_recording, si.BaseRecording) and self._segments is not None
        if not use_cached:
            self._raw_si_recording, self._segments = self._wneSubject.get_si_recording(
                self._experiment,
                self._alias,
                "ap",
                self._probe,
                combine="concatenate",
                exclusions=self._exclusions.copy(),  # Avoid in-place modification so we check consistency
                sampling_frequency_max_diff=1e-06,
            )
        # Ensure we used the same segments as previously
        prior_segments_path = self.main_output_dir / SEGMENTS_FNAME
        if prior_segments_path.exists():
            prior_segments = utils.read_htsv(prior_segments_path)
            try:
                # Couldn't make it work for all columns because of rounding x dtype
                cols_to_compare = [
                    "fname",
                    "type",
                    "imSampRate",
                    "start_frame",
                    "end_frame",
                    "fileDuration",
                    "segmentDuration",
                ]
                assert_frame_equal(
                    prior_segments[cols_to_compare], self._segments[cols_to_compare]
                )
            except AssertionError as e:
                raise ValueError(
                    f"Newly computed segments don't match previously used segment file at {prior_segments_path}:\n"
                    f"Consider instantiating pipeline object with `SpikeinterfaceSortingPipeline.load_from_folder`\n"
                    f"{e}"
                )
        return self._raw_si_recording, self._segments

    @property
    def spikeinterface_output_dir(self) -> Path:
        # Differs from main_output_dir because spikeinterface deletes everything in there when running a sorting.
        return self.main_output_dir / "si_output"

    @property
    def sorter_output_dir(self) -> Path:
        return self.spikeinterface_output_dir / "sorter_output"

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
        return self._preprocessed_probe

    @property
    def preprocessing_output_dir(self) -> Path:
        return self.main_output_dir / "preprocessing"

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
        self.main_output_dir.mkdir(exist_ok=True, parents=True)
        with open(self.main_output_dir / OPTS_FNAME, "w") as f:
            yaml.dump(self.opts, f)
        utils.write_htsv(self._exclusions, self.main_output_dir / EXCLUSIONS_FNAME)
        utils.write_htsv(segments, self.main_output_dir / SEGMENTS_FNAME)

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
            self.main_output_dir / OPTS_FNAME,
            self.main_output_dir / "preprocessed_si_probe.json",
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

        if self._preprocessed_si_recording is None:
            raise AttributeError(
                f"No preprocessing object found.\n" "You need to run preprocessing."
            )

        # Get sorter and parameters
        sorting_opts = self.opts["sorting"]
        sorter_name = sorting_opts["sorter_name"]
        sorter_params = sorting_opts["sorter_params"]

        # If using KiloSort2.5, we need to set the path to the KiloSort executable
        if sorter_name == "kilosort2_5":
            if not "sorter_path" in sorting_opts:
                raise ValueError(
                    "Expected 'sorter_path' key in sorting opts for kilosort.\n"
                    "You might be using obsolete formatting for opts file?"
                )
            ss.sorter_dict[sorter_name].set_kilosort2_5_path(
                sorting_opts["sorter_path"]
            )
        else:
            raise NotImplementedError()

        # Sort
        assert self.main_output_dir.exists()  # Created during prepro
        with Timing(name="Run spikeinterface sorter"):
            ss.run_sorter(
                sorter_name,
                self._preprocessed_si_recording,
                output_folder=self.spikeinterface_output_dir,
                verbose=True,
                with_output=False,
                **sorter_params,
                **self.job_kwargs,
            )

        # Options should be have been saved already during preprocessing
        assert (self.main_output_dir / OPTS_FNAME).exists()
        assert (self.main_output_dir / EXCLUSIONS_FNAME).exists()
        assert (self.main_output_dir / SEGMENTS_FNAME).exists()

    ##### Sorting output #####

    def get_kilosort_binary_recording_extractor(self) -> si.BinaryRecordingExtractor:
        if self._kilosort_binary_recording_extractor is None:
            self._kilosort_binary_recording_extractor = (
                utils.siutils.load_kilosort_bin_as_si_recording(
                    self.sorter_output_dir,
                    fname=self.preprocessed_bin_path.name,
                    si_probe=self.preprocessed_probe,
                )
            )
        return self._kilosort_binary_recording_extractor

    def get_processed_extractor_for_waveforms(self) -> si.BaseRecording:
        """Return binary recording.dat if available, or repreprocess lazily"""
        if self.preprocessed_bin_path.exists():  # No need to re-run
            return self.get_kilosort_binary_recording_extractor()
        else:  # Need to re-run.
            if self._preprocessed_si_recording is None:
                self.run_preprocessing()
            return self._preprocessed_si_recording

    def get_sorting_extractor(self, with_recording=True) -> si.KiloSortSortingExtractor:
        if self._si_sorting_extractor is None:
            self._si_sorting_extractor = se.read_kilosort(self.sorter_output_dir)
        sorting = self._si_sorting_extractor
        if with_recording and not sorting.has_recording():
            sorting.register_recording(self.get_processed_extractor_for_waveforms())
        return sorting

    ##### Reinstantiate ######

    @classmethod
    def load_from_folder(
        cls,
        wneProject: Project,
        wneSubject: SGLXSubject,
        experiment: str,
        alias: str,
        probe: str,
        basename: str,
        rerun_existing: bool = False,
        n_jobs: int = 1,
    ):
        """Instantiate a pipeline object from previous run."""

        main_output_dir = get_main_output_dir(
            wneProject,
            wneSubject,
            experiment,
            alias,
            probe,
            basename,
        )

        if not all(
            [
                f.exists()
                for f in [
                    main_output_dir,
                    main_output_dir / OPTS_FNAME,
                    main_output_dir / EXCLUSIONS_FNAME,
                    main_output_dir / SEGMENTS_FNAME,
                ]
            ]
        ):
            raise FileNotFoundError(
                f"Could not find all required files in {main_output_dir}"
            )

        return SpikeInterfaceSortingPipeline(
            wneProject,
            wneSubject,
            experiment,
            alias,
            probe,
            basename,
            rerun_existing=rerun_existing,
            n_jobs=n_jobs,
            options_source=(main_output_dir / OPTS_FNAME),
            exclusions_source=(main_output_dir / EXCLUSIONS_FNAME),
        )
