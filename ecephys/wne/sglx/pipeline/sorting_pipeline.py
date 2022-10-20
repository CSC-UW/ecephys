import logging
from ecephys import wne
from . import si_utils


logger = logging.getLogger(__name__)

class AbstractSortingPipeline:

    def __init__(
        self,
        subjectName,
        subjectsDir,
        experimentName,
        projectName,
        projectsFile,
        alias,
    ):
        # Project and data
        subjLib = wne.sglx.SubjectLibrary(subjectsDir)
        projLib = wne.ProjectLibrary(projectsFile)
        self.subj = subjLib.get_subject(subjectName)
        self.proj = projLib.get_project(projectName)
        self.alias = alias

        # Pipeline options
        self.opts = self.proj.load_experiment_subject_json(
            experimentName, self.subj.name, wne.constants.EXP_PARAMS_FNAME
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

        # Input, output files and folders
        # All as properties in case we change project for SortingPipeline object
        # And because all preprocessing is done in a lazy way anyways
        self._raw_ap_bin_table = None
        self._raw_si_recording =  None
        self._processed_si_recording = None
        self._processed_bin_dump_path = None
        self._raw_sorting_output_dir = None
        self._postprocessed_sorting_output_dir = None
        self._final_sorting_output_dir = None  # With metrics and waveforms

        # Pipeline steps
        self._is_preprocessed = False
        self._is_dumped = False
        self._is_sorted = False
        self._is_postprocessed = False
        self._is_curated = False
        self._is_run_metrics = False

    ######### Properties ##########

    @property
    def raw_ap_bin_table(self):
        # TODO
        pass
    
    @property
    def raw_ap_binpaths(self):
        # TODO
        pass

    @property
    def raw_si_recording(self):
        self.load_raw_si_recording()

    @property
    def processed_si_recording(self):
        self.preprocess_raw_si_recording()

    @property
    def processed_bin_dump_path(self):
        # TODO
        pass

    @property
    def raw_sorting_output_dir(self):
        # TODO
        pass

    @property
    def postprocessed_sorting_output_dir(self):
        # TODO
        pass

    @property
    def final_sorting_output_dir(self):
        # TODO
        pass

    ######### Pipeline steps ##########

    def load_raw_si_recording(self):
        return si_utils.load_si_recording(
            self.ap_binpaths,
            self.opts,
        )

    def preprocess_raw_si_recording(self):
        assert self.raw_si_recording is not None
        self.processed_si_recording = si_utils.preprocess_si_recording(
            self.raw_si_recording,
            self.opts,
        )
        self._is_preprocessed = True

    def dump_preprocessed_si_recording(self):
        if not self._is_preprocessed:
            raise ValueError("Preprocess before dump")
        # TODO
        self._is_dumped = True
    
    def run_sorting(self):
        if not self._is_dumped:
            raise ValueError("Dump before sorting")
        # TODO
        self._is_sorted = True

    def run_postprocessing(self):
        if not self._is_dumped:
            raise ValueError("Sort before postprocessing")
        # TODO
        self._is_postprocessed = True

    def run_metrics(self):
        if not self._is_curated:
            raise ValueError("Curate before running metrics")
        # TODO
        self._is_run_metrics = True


def load_from_pickle(path):
    # TODO
    pass