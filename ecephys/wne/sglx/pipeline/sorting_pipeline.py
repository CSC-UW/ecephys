import logging
from pathlib import Path

import probeinterface as pi
import yaml
from horology import Timing, timed

import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.sorters as ss
from ecephys import wne
from ecephys.utils import spikeinterface_utils as si_utils

from .preprocess_si_rec import preprocess_si_recording

logger = logging.getLogger(__name__)

class AbstractSortingPipeline:

    def __init__(
        self,
        subjectName,
        subjectsDir,
        projectName,
        projectsFile,
        experimentName,
        aliasName,
        probe,
        time_ranges=None,
        opts_filepath=None,
        output_dirname=None,
        rerun_existing=True,
    ):
        # Project and data
        subjLib = wne.sglx.SubjectLibrary(subjectsDir)
        projLib = wne.ProjectLibrary(projectsFile)
        self.wneSubject = subjLib.get_subject(subjectName)
        self.wneProject = projLib.get_project(projectName)
        self.aliasName = aliasName
        self.experimentName = experimentName
        self.probe = probe
        self.time_ranges = time_ranges
        if self.time_ranges is not None:
            if not len(self.time_ranges) == len(self.raw_ap_bin_table):
                raise ValueError(
                    f"`time_ranges` should be a list of length "
                    f"{len(self.raw_ap_bin_table)}"
                )

        # Pipeline options
        if opts_filepath is None:
            self.opts = self.wneProject.load_experiment_subject_json(
                experimentName,
                self.wneSubject.name,
                wne.constants.SORTING_PIPELINE_PARAMS_FNAME,
            )
        else:
            opts_filepath = Path(opts_filepath)
            assert opts_filepath.exists(), f"{opts_filepath}"
            with open(opts_filepath, 'r') as f:
                self.opts = yaml.load(f, Loader=yaml.SafeLoader)
        self.rerun_existing = rerun_existing

        # Input/Output
        self._raw_ap_bin_table = None
        self._output_dirname = output_dirname
        if self._output_dirname is None:
            if not "output_dirname" in self.opts:
                raise ValueError(
                    "Specify 'output_dirname' as kwarg or under 'output_dirname' key in opts file."
                )
            self._output_dirname = self.opts["output_dirname"]
        self._output_dirname = f"{self._output_dirname}.{self.probe}"
        self._sorting_output_dirname = self._output_dirname
        self._preprocessing_output_dirname = f"prepro_{self._output_dirname}"
        self._output_dir = None

    def __repr__(self):
        return f"""
        {self.__class__.__name__}:
        {self.wneSubject.name}, {self.probe}
        {self.experimentName} - {self.aliasName}
        Segment time ranges (s): {self.time_ranges}
        Sorting output_dir: {self.sorting_output_dir}
        Preprocessing output_dir: {self.preprocessing_output_dir}
        First file full path: \n{self.raw_ap_bin_table.path.values[0]}
        AP table: \n{self.raw_ap_bin_table}
        """

    ######### Input output ####################

    @property
    def output_dir(self):
        return self.wneProject.get_alias_subject_directory(
            self.experimentName,
            self.aliasName,
            self.wneSubject.name
        )/self._output_dirname

    @property
    def sorting_output_dir(self):
        return self.output_dir

    @property
    def preprocessing_output_dir(self):
        return self.wneProject.get_alias_subject_directory(
            self.experimentName,
            self.aliasName,
            self.wneSubject.name
        )/self._preprocessing_output_dirname

    @property
    def raw_ap_bin_table(self):
        return self.wneSubject.get_ap_bin_table(
            self.experimentName,
            alias=self.aliasName,
            probe=self.probe,
        )

    ######### Dump / load full pipeline ##########

    def to_pickle():
        # TODO
        pass

    ######### Utils ##########

    def switch_project(newProjectName):
        raise NotImplementedError()

    ######### Pipeline steps ##########

    def run_pipeline():
        raise NotImplementedError()

    def run_metrics():
        raise NotImplementedError()


# All relying on spikeinterface
class SpikeInterfaceSortingPipeline(AbstractSortingPipeline):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Input, output files and folders, specific to SI
        # All as properties in case we change project for SortingPipeline object
        # And because all preprocessing is done in a lazy way anyways
        self._raw_si_recording =  None
        self._processed_si_recording = None
        self._dumped_bin_si_recording = None
        self._sorting_extractor = None
        self._waveform_extractor = None

        # Pipeline steps, specific to SI
        self._is_preprocessed = False
        self._is_dumped = False
        self._is_sorted = False
        self._is_postprocessed = False
        self._is_curated = False
        self._is_run_metrics = False

    ######### Paths ##########

    @property
    def processed_bin_path(self):
        return self.sorting_output_dir/'recording.dat'

    @property
    def processed_si_probe_path(self):
        return self.sorting_output_dir/'processed_si_probe.json'

    @property
    def waveforms_dir(self):
        return self.sorting_output_dir/'waveforms'

    ######### SI objects ##########

    @property
    def raw_si_recording(self):
        if self._raw_si_recording is None:
            self._raw_si_recording = self.load_raw_si_recording()
        return self._raw_si_recording
    
    @property
    def processed_si_recording(self):
        if self._processed_si_recording is None:
            self.run_preprocessing()
        return self._processed_si_recording

    @property
    def dumped_bin_si_recording(self):
        if self._dumped_bin_si_recording is None:
            self._dumped_bin_si_recording = self.load_dumped_bin_si_recording()
        return self._dumped_bin_si_recording

    @property
    def processed_si_probe(self):
        return self.processed_si_recording.get_probe()
    
    @property
    def sorting_extractor(self):
        if self._sorting_extractor is None:
            self._sorting_extractor = self.load_sorting_extractor()
        return self._sorting_extractor

    @property
    def si_waveform_extractor(self):
        if self._waveform_extractor is None:
            self._waveform_extractro = self.load_waveform_extractor()
        return self._waveform_extractor

    ######### IO ##########

    def load_raw_si_recording(self):
        # return self.wneSubject.get_multi_segment_si_recording(
        return self.wneSubject.get_single_segment_si_recording(
            self.experimentName,
            self.aliasName,
            'ap',
            self.probe,
            time_ranges=self.time_ranges,
            sampling_frequency_max_diff=1e-06,
        )

    def load_sorting_extractor(self):
        return se.read_kilosort(self.sorting_output_dir) 

    def load_dumped_bin_si_recording(self):
        return si_utils.load_kilosort_bin_as_si_recording(
            self.sorting_output_dir,
            fname=processed_bin_path.name,
            si_probe=self.load_si_probe,
        )

    def load_waveform_extractor(self):
        return si.extract_waveforms(
            self.dumped_bin_si_recording,
            self.sorting_extractor,
            folder=self.waveforms_dir,
            load_if_exists=True,
        )

    def dump_si_probe(self):
        # Used to load probe when running waveform extractor/quality metrics
        # NB: The probe used for sorting may differ from the raw SGLX probe,
        # Since somme channels may be dropped during motion correction
        self.processed_si_probe_path.parent.mkdir(exist_ok=True, parents=True)
        pi.write_probeinterface(
            self.processed_si_probe_path,
            self.processed_si_probe,
        )
        print(f"Save processed SI probe at {self.processed_si_probe_path}")

    def load_si_probe(self):
        # Used to load probe when running waveform extractor/quality metrics
        # NB: The probe used for sorting may differ from the raw SGLX probe,
        # Since somme channels may be dropped during motion correction
        if not self.processed_si_probe_path.exists():
            raise FileNotFoundError(
                f"Could not find spikeinterface probe object at {self.processed_si_probe_path}"
            )
        prb_grp = pi.read_probeinterface(self.processed_si_probe_path)
        assert len(prb_grp.probes) == 1
        return prb_grp.probes[0]

    def dump_opts(self, output_dir=None):
        if output_dir is None:
            output_dir = self.output_dir
        opts_to_dump = self.opts.copy()
        opts_to_dump['time_ranges'] = self.time_ranges
        with open(output_dir/"sorting_pipeline_opts.yaml", 'w') as f:
            yaml.dump(opts_to_dump, f)

    ######### Pipeline steps ##########

    def run_preprocessing(self):
        assert self.raw_si_recording is not None
        self._processed_si_recording = preprocess_si_recording(
            self.raw_si_recording,
            self.opts,
            output_dir=self.preprocessing_output_dir,
            rerun_existing=self.rerun_existing,
        )

        # Save if opts if we did some preprocessing
        if self.preprocessing_output_dir.exists():
            self.dump_opts(self.preprocessing_output_dir)

        self._is_preprocessed = True

    def run_sorting(self):

        if (self.sorting_output_dir / "spike_times.npy").exists() and not self.rerun_existing:
            print(f"Passing: output directory is already done: {self.sorting_output_dir}\n\n")
            return True

        sorter_name, sorter_params = self.opts["sorting"]

        self.sorting_output_dir.mkdir(exist_ok=True, parents=True)
        if sorter_name == "kilosort2_5":
            sorter_params = sorter_params.copy() # Allow rerunning since we pop
            ss.sorter_dict[sorter_name].set_kilosort2_5_path(sorter_params.pop("ks_path"))

        with Timing(name="Run spikeinterface sorter"):
            ss.run_sorter(
                sorter_name,
                self.processed_si_recording,
                output_folder=self.sorting_output_dir,
                verbose=True,
                with_output=False,
                **sorter_params,
            )

        self.dump_si_probe()
        self.dump_opts(self.sorting_output_dir)

        self._is_sorted = True

    def run_drift_estimates(self):
        self.run_preprocessing()
        self.dump_opts()

    def run_pipeline(self):
        self.run_preprocessing()
        self.run_sorting()

    def run_postprocessing(self):
        raise NotImplementedError()

    def run_metrics(self):
        print(f"Running metrics with params {metrics_opts}")

        # Extract joblib params
        # Get list of metrics
        # get set of params across metrics
        metrics_opts = self.opts['metrics']['metrics_names'].copy()
        multipro_params = metrics_opts.pop('_multipro_params', {})
        metrics_names = list(metrics_opts.keys())
        params = {
            **metric_dict
            for metric_dict in metrics_opts.values()
        }

        metrics = sq.compute_quality_metrics(
            waveform_extractor,
            metric_names=metrics_names,
            **params,
            **multipro_params
        )

        print("Save metrics dataframe as `metrics.csv` in kilosort dir")
        metrics_df.to_csv(
            ks_output_dir/"metrics.csv",
            # sep="\t",
            index=True,
            index_label="cluster_id"
        )
        print("Done")


class SortingPipeline(SpikeInterfaceSortingPipeline):
    
    def __init__(*args, **kwargs):
        super().__init__(*args, **kwargs)


def load_from_pickle(path):
    # TODO
    pass
