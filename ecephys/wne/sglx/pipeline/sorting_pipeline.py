import logging
from pathlib import Path

import pandas as pd
import probeinterface as pi
import yaml
from horology import Timing, timed

import spikeinterface.extractors as se
import spikeinterface.full as si
import spikeinterface.postprocessing as sp
import spikeinterface.qualitymetrics as sq
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
        artifacts_filepath=None,
        output_dirname=None,
        rerun_existing=True,
        n_jobs=1,
    ):
        # Project and data
        subjLib = wne.sglx.SubjectLibrary(subjectsDir)
        projLib = wne.ProjectLibrary(projectsFile)
        self.wneSubject = subjLib.get_subject(subjectName)
        self.wneProject = projLib.get_project(projectName)
        self.aliasName = aliasName
        self.experimentName = experimentName
        self.probe = probe

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
        self.n_jobs = int(n_jobs)

        # Output
        self._raw_ap_bin_segment_frame = None
        self._output_dirname = output_dirname
        if self._output_dirname is None:
            raise ValueError("Please specify 'output_dirname' kwarg")
        self._output_dirname = f"{self._output_dirname}.{self.probe}"
        self._preprocessing_output_dirname = f"prepro_{self._output_dirname}"
        self._output_dir = None

        # Sub-segments
        if self.output_artifacts_filepath.exists():
            if artifacts_filepath is not None:
                raise ValueError(
                    f"Some artifacts were already provided for this sorting, please don't set 'artifacts_filepath' kwarg."
                    f" Artifacts file saved at {self.output_artifacts_filepath} will be used."
                )
            artifacts_filepath = self.output_artifacts_filepath
        self.artifacts_frame=self.load_artifacts_frame(artifacts_filepath)
        self.time_ranges = time_ranges
        if self.time_ranges is not None:
            if not len(self.time_ranges) == len(self.raw_ap_bin_segment_frame):
                raise ValueError(
                    f"`time_ranges` should be a list of length "
                    f"{len(self.raw_ap_bin_segment_frame)}"
                )
        assert self.time_ranges is None or self.artifacts_frame is None


    def __repr__(self):
        return f"""
        {self.__class__.__name__}:
        {self.wneSubject.name}, {self.probe}
        {self.experimentName} - {self.aliasName}
        N_jobs: {self.n_jobs}
        Segment time ranges (s): {self.time_ranges}
        Sorting output_dir: {self.sorting_output_dir}
        Preprocessing output_dir: {self.preprocessing_output_dir}
        First segment full path: \n{self.raw_ap_bin_segment_frame.path.values[0]}
        Artifacts: {self.artifacts_frame}
        Total sorting duration: \n{self.raw_si_recording.get_total_duration()}(s)
        AP segment table: \n{self.raw_ap_bin_segment_frame.loc[:,['run', 'gate', 'trigger', 'probe', 'wneSegmentStartTime', 'segment_idx', 'segmentTimeSecs', 'segmentFileSizeRatio']]}
        """


    ######### Data

    @property
    def raw_ap_bin_segment_frame(self):
        return self.wneSubject.get_segment_frame(
            self.experimentName,
            aliasName=self.aliasName,
            probe=self.probe,
            ftype="bin",
            stream="ap",
            artifacts_frame=self.artifacts_frame,
        )

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
        return self.output_dir/'sorter_output'

    @property
    def preprocessing_output_dir(self):
        return self.wneProject.get_alias_subject_directory(
            self.experimentName,
            self.aliasName,
            self.wneSubject.name
        )/self._preprocessing_output_dirname

    @property
    def output_artifacts_filepath(self):
        return self.output_dir/"artifacts.{self.probe}.tsv}"

    def dump_segment_frame(self):
        self.raw_ap_bin_segment_frame.to_csv(
            self.output_dir/"segments_frame.tsv",
            sep="\t"
        )

    def dump_artifacts_frame(self):
        if self.artifacts_frame is None:
            return
        self.artifacts_frame.to_csv(
            self.output_artifacts_filepath,
            sep="\t",
        )

    def load_artifacts_frame(self, fpath):
        if fpath is None:
            return None
        fpath = Path(fpath)
        assert fpath.name.endswith(".tsv")
        assert fpath.exists()
        df = pd.read_csv(fpath, sep='\t')
        assert set(df.columns) == set(["start", "stop", "file"])
        return df

    def dump_opts(self, output_dir=None, fname="sorting_pipeline_opts.yaml"):
        if output_dir is None:
            output_dir = self.output_dir
        opts_to_dump = self.opts.copy()
        opts_to_dump['time_ranges'] = self.time_ranges
        with open(output_dir/fname, 'w') as f:
            yaml.dump(opts_to_dump, f)
    
    def load_opts(self, output_dir=None, fname="sorting_pipeline_opts.yaml"):
        if output_dir is None:
            output_dir = self.output_dir
        with open(output_dir/fname, 'r') as f:
            return yaml.load(f, Loader=yaml.SafeLoader)


    ######### Pipeline steps ####################
    
    @property
    def is_sorted(self):
        raise NotImplementedError()

    @property
    def is_postprocessed(self):
        raise NotImplementedError()

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
        self._si_sorting_extractor = None
        self._si_waveform_extractor = None

        # TODO use only properties
        # Pipeline steps, specific to SI
        self._is_preprocessed = False
        self._is_dumped = False
        self._is_sorted = False
        self._is_postprocessed = False
        self._is_curated = False
        self._is_run_metrics = False

        self.job_kwargs = {
            "n_jobs": self.n_jobs,
            "chunk_duration": "1s",
            "progress_bar": True,
        }

    ######### Paths ##########

    @property
    def processed_bin_path(self):
        return self.sorting_output_dir/'recording.dat'

    @property
    def processed_si_probe_path(self):
        return self.output_dir/'processed_si_probe.json'

    @property
    def waveforms_dir(self):
        return self.output_dir/'waveforms'
    

    ######### Pipeline steps ######
    
    def check_finished_sorting(self):
        required_output_files = [
            self.sorting_output_dir/"amplitudes.npy",
            self.output_dir/"sorting_pipeline_opts.yaml",
            self.output_dir/"processed_si_probe.json",
        ]
        if not all([f.exists() for f in required_output_files]):
            raise FileNotFoundError(
                "Sorting seems to be finished: expected to find all of the following files: \n"
                f"{required_output_files}"
            )

    def check_finished_postprocessing(self):
        #TODO
        pass

    @property
    def is_sorted(self):
        sorted = (
            self.sorting_output_dir.exists()
            and (self.sorting_output_dir/"spike_times.npy").exists()
        )
        if sorted:
            self.check_finished_sorting()
        return sorted

    @property
    def is_postprocessed(self):
        postprocessed = (
            self.is_sorted
            and self.waveforms_dir.exists()
            and (self.sorting_output_dir/"metrics.csv").exists()
        )
        if postprocessed:
            self.check_finished_postprocessing()
        return postprocessed

    ######### SI objects ##########

    @property
    def raw_si_recording(self):
        if self._raw_si_recording is None:
            self._raw_si_recording = self.load_raw_si_recording()
            if not self.time_ranges:
                # assert self._raw_si_recording.get_total_duration() == self.raw_ap_bin_segment_frame.segmentTimeSecs.astype(float).sum()
                assert self._raw_si_recording.get_total_samples() == self.raw_ap_bin_segment_frame.nSegmentSamp.astype(int).sum()
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

    def load_raw_si_recording(self):
        # return self.wneSubject.get_multi_segment_si_recording(
        return self.wneSubject.get_single_segment_si_recording(
            self.experimentName,
            self.aliasName,
            'ap',
            self.probe,
            time_ranges=self.time_ranges,
            artifacts_frame=self.artifacts_frame,
            sampling_frequency_max_diff=1e-06,
        )

    def load_si_sorting_extractor(self):
        return se.read_kilosort(self.sorting_output_dir) 

    def load_dumped_bin_si_recording(self):
        return si_utils.load_kilosort_bin_as_si_recording(
            self.sorting_output_dir,
            fname=self.processed_bin_path.name,
            si_probe=self.load_si_probe(),
        )

    def load_si_waveform_extractor(self):
        MS_BEFORE=1.5
        MS_AFTER=2.5
        MAX_SPIKES_PER_UNIT=2000
        SPARSITY_RADIUS=400
        NUM_SPIKES_FOR_SPARSITY=500

        assert self.is_sorted
        load_precomputed = (
            not self.rerun_existing 
            and self.is_postprocessed
        )
        if self.processed_bin_path.exists():
            waveform_recording = self.dumped_bin_si_recording
        else:
            # Make sure we preprocess in the same way as during sorting
            waveform_recording = self.run_preprocessing(load_existing=True)

        if load_precomputed:
            print(f"Loading waveforms from folder. Will use this recording: {waveform_recording}")
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

    def dump_si_probe(self):
        # Used to load probe when running waveform extractor/quality metrics
        # NB: The probe used for sorting may differ from the raw SGLX probe,
        # Since somme channels may be dropped during motion correction
        # self.processed_si_probe_path.parent.mkdir(exist_ok=True, parents=True)
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
            import warnings
            warnings.warn(
            # raise FileNotFoundError(
                f"Find no spikeinterface probe object at {self.processed_si_probe_path}"
                f"\nRe-dumping probe"
            )
            self.dump_si_probe()
        prb_grp = pi.read_probeinterface(self.processed_si_probe_path)
        assert len(prb_grp.probes) == 1
        prb = prb_grp.probes[0]
        print(prb)
        return prb

    ######### Pipeline steps ##########

    def run_preprocessing(self, load_existing=False):
        assert self.raw_si_recording is not None

        if load_existing:
            # Ensures we use the same opts as previously during postprocessing
            print("Preprocessing: load_existing=True: Use preexisting opts! Ignore provided preprocessing opts")
            assert self.is_sorted
            prepro_opts = self.load_opts()
        else:
            prepro_opts = self.opts

        self._processed_si_recording = preprocess_si_recording(
            self.raw_si_recording,
            prepro_opts,
            output_dir=self.preprocessing_output_dir,
            rerun_existing=self.rerun_existing,
            job_kwargs=self.job_kwargs,
        )

        # Save if opts if we did some preprocessing
        if self.preprocessing_output_dir.exists():
            self.dump_opts(self.preprocessing_output_dir)

        self._is_preprocessed = True

        return self._processed_si_recording

    def run_sorting(self):

        if self.is_sorted and not self.rerun_existing:
            print(f"Passing: output directory is already done: {self.sorting_output_dir}\n\n")
            return True

        sorter_name, sorter_params = self.opts["sorting"]

        self.output_dir.mkdir(exist_ok=True, parents=True)
        if sorter_name == "kilosort2_5":
            sorter_params = sorter_params.copy() # Allow rerunning since we pop
            ss.sorter_dict[sorter_name].set_kilosort2_5_path(sorter_params.pop("ks_path"))

        with Timing(name="Run spikeinterface sorter"):
            ss.run_sorter(
                sorter_name,
                self.processed_si_recording,
                output_folder=self.output_dir,
                verbose=True,
                with_output=False,
                **sorter_params,
                **self.job_kwargs,
            )

        self.dump_si_probe()
        self.dump_opts()

        self._is_sorted = True

    def run_drift_estimates(self):
        self.run_preprocessing()
        self.dump_opts()

    def run_pipeline(self):
        self.run_preprocessing()
        self.run_sorting()

    def run_postprocessing(self):

        UNIT_LOCATIONS_METHOD = "center_of_mass"

        with Timing(name="Compute principal components: "):
            print("Computing principal components.")
            sp.compute_principal_components(
                self.si_waveform_extractor,
                load_if_exists=True,
                n_components=5,
                mode="by_channel_local",
                n_jobs=self.n_jobs,
            )

        # with Timing(name="Compute spike amplitudes: "):
        #     print("Computing spike amplitudes.")
        #     sp.compute_spike_amplitudes(
        #         self.si_waveform_extractor,
        #         load_if_exists=True,
        #         n_jobs=self.n_jobs,
        #     )

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

    def run_metrics(self):

        # Get list of metrics
        # get set of params across metrics
        metrics_opts = self.opts['metrics'].copy()
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
                n_jobs=self.n_jobs,
                progress_bar=True,
                verbose=True,
                **params,
            )

        print("Save metrics dataframe as `metrics.csv` in kilosort dir")
        metrics_df.to_csv(
            self.sorting_output_dir/"metrics.csv",
            # sep="\t",
            index=True,
            index_label="cluster_id"
        )

        self.dump_opts(self.output_dir, fname="metrics_opts.yaml")
        print("Done")


class SortingPipeline(SpikeInterfaceSortingPipeline):
    
    def __init__(*args, **kwargs):
        super().__init__(*args, **kwargs)


def load_from_pickle(path):
    # TODO
    pass
