"""
WNE project organization:

- project_dir/ (e.g. SPWRs/)
  - subject_dir/ (e.g. ANPIX11-Adrian/)
  - experiment_dir/ (e.g. novel_objects_deprivation/)
    - alias_dir/ (e.g. recovery_sleep/)
      - alias_subject_dir/ (e.g. ANPIX11-Adrian/)
    - subalias_dir/ (e.g. sleep_homeostasis_0/, sleep_homeostasis_1/) <- Only if there is N>1 subaliases
      - subalias_subject_dir/ (e.g. ANPIX11-Adrian/)
    - experiment_subject_dir/ (e.g. ANPIX11-Adrian/)


Example projects file (YAML format):

---
project: my_project
project_directory: /path/to/project/
...
---
project: my_other_project
project_directory: /path/to/other_project
...

"""

import json
import logging
import pickle
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
# Tom's notes:
# Experiments + aliases assumed, sessions not?
# Only projects.yaml is required to resolve paths?
# You could name a project the same thing as an experiment
# You could name a project "Common" or "Scoring" or "Sorting"
import pandas as pd
import yaml

import spikeinterface.extractors as se
from ecephys import hypnogram
from ecephys.units.si_ks_sorting import SpikeInterfaceKilosortSorting

from .. import sglx as ece_sglx
from .. import sharptrack, sync, units
from .. import utils as ece_utils
from . import constants
from .sglx import sessions

Pathlike = Union[Path, str]

logger = logging.getLogger(__name__)


def load_yaml_stream(yaml_path: Pathlike) -> list[dict]:
    """Load all YAML documents in a file."""
    with open(yaml_path) as fp:
        yaml_stream = list(yaml.safe_load_all(fp))
    return yaml_stream


class Project:
    def __init__(self, project_name: str, project_dir: Pathlike):
        self.name = project_name
        self.dir = Path(project_dir)

    def __repr__(self):
        return f"{self.name}: {self.dir}"

    #####
    # Methods for getting directories
    #####

    def get_subject_directory(self, subject: str) -> Path:
        """Get a subject's directory for this project."""
        return self.dir / subject

    def get_experiment_directory(self, experiment: str) -> Path:
        return self.dir / experiment

    def get_alias_directory(self, experiment: str, alias: str) -> Path:
        return self.get_experiment_directory(experiment) / alias

    def get_subalias_directory(
        self, experiment: str, alias: str, subalias_idx: int
    ) -> Path:
        return self.get_experiment_directory(experiment) / f"{alias}_{subalias_idx}"

    def get_experiment_subject_directory(self, experiment: str, subject: str) -> Path:
        return self.get_experiment_directory(experiment) / subject

    def get_alias_subject_directory(
        self, experiment: str, alias: str, subject: str
    ) -> Path:
        return self.get_alias_directory(experiment, alias) / subject

    def get_alias_subject_directory(
        self, experiment: str, alias: str, subject: str
    ) -> Path:
        return self.get_alias_directory(experiment, alias) / subject

    def get_subalias_subject_directory(
        self, experiment: str, alias: str, subject: str, subalias_idx: int
    ) -> Path:
        return self.get_subalias_directory(experiment, alias, subalias_idx) / subject

    #####
    # Methods for getting files
    #####

    def get_project_file(self, fname: str) -> Path:
        return self.dir / fname

    def get_project_subject_file(self, subject: str, fname: str) -> Path:
        return self.get_subject_directory(subject) / fname

    def get_experiment_file(self, experiment: str, fname: str) -> Path:
        return self.get_experiment_directory(experiment) / fname

    def get_alias_file(
        self,
        experiment: str,
        alias: str,
        fname: str,
        subalias_idx: Optional[int] = None,
    ) -> Path:
        return (
            self.get_alias_directory(experiment, alias, subalias_idx=subalias_idx)
            / fname
        )

    def get_experiment_subject_file(
        self, experiment: str, subject: str, fname: str
    ) -> Path:
        return self.get_experiment_subject_directory(experiment, subject) / fname

    def get_alias_subject_file(
        self, experiment: str, alias: str, subject: str, fname: str
    ) -> Path:
        return self.get_alias_subject_directory(experiment, alias, subject) / fname

    # TODO: This should probably be a function in ece.wne.utils, since projects should be agnostic to SGLX vs TDT etc.
    def get_sglx_counterparts(
        self,
        subject: str,
        paths: list[Pathlike],
        extension: str,
        remove_probe: bool = False,
        remove_stream: bool = False,
    ) -> list[Path]:
        """Get counterparts to SpikeGLX raw data files.

        Counterparts are mirrored at the project's subject directory, and likely
        have different suffixes than the original raw data files.

        Parameters:
        -----------
        project_name: str
            From projects.yaml
        subject_name: str
            Subject's name within this project, i.e. subject's directory name.
        paths: list of pathlib.Path
            The raw data files to get the counterparts of.
        extension:
            The extension to replace .bin or .meta with. See `replace_ftype`.

        Returns:
        --------
        list of pathlib.Path
        """
        counterparts = sessions.mirror_raw_data_paths(
            self.get_subject_directory(subject), paths
        )  # Mirror paths at the project's subject directory
        counterparts = [
            ece_sglx.file_mgmt.replace_ftype(p, extension, remove_probe, remove_stream)
            for p in counterparts
        ]
        return ece_utils.remove_duplicates(counterparts)

    def load_experiment_subject_json(
        self, experiment: str, subject: str, fname: str
    ) -> dict:
        path = self.get_experiment_subject_file(experiment, subject, fname)
        with open(path) as f:
            return json.load(f)

    def load_experiment_subject_yaml(
        self, experiment: str, subject: str, fname: str
    ) -> dict:
        path = self.get_experiment_subject_file(experiment, subject, fname)
        with open(path) as f:
            return yaml.load(f, Loader=yaml.SafeLoader)

    def get_all_probes(self, subject: str, experiment: str) -> list[str]:
        opts = self.load_experiment_subject_json(
            experiment, subject, constants.EXP_PARAMS_FNAME
        )
        return list(opts["probes"].keys())

    def get_main_sorting_dir(
        self, subject: str, experiment: str, alias: str, probe: str, sorting: str
    ):
        return (
            self.get_alias_subject_directory(experiment, alias, subject)
            / f"{sorting}.{probe}"
        )

    # TODO: If no sortingName is provided, a sensible default should be obtained from WNE opts, so that you do not have do remember the sorting ID for every animal.
    def get_kilosort_extractor(
        self, subject: str, experiment: str, alias: str, probe: str, sorting: str, postprocessing: str = None,
    ) -> se.KiloSortSortingExtractor:
        """Load the contents of a Kilosort output directory. This takes ~20-25s per 100 clusters."""
        main_sorting_dir = self.get_main_sorting_dir(
            subject, experiment, alias, probe, sorting
        )
        sorter_output_dir = main_sorting_dir / "si_output/sorter_output"
        assert sorter_output_dir.is_dir(), f"Expected Kilosort directory not found: {sorter_output_dir}"
        assert (sorter_output_dir/"spike_times.npy").exists(), f"Expected `spike_times.npy` file in: {sorter_output_dir}"
        extractor = se.read_kilosort(sorter_output_dir, keep_good_only=False)

        if postprocessing is not None:
            # Load metrics and add as properties
            postprocessing_dir = main_sorting_dir / postprocessing
            metrics_path = postprocessing_dir / "metrics.csv"
            assert postprocessing_dir.is_dir(), f"Expected postprocessing directory not found: {postprocessing_dir}"
            assert metrics_path.exists(), f"Expected `metrics.csv` file in: {postprocessing_dir}"

            metrics = pd.read_csv(metrics_path)
            assert all([c in metrics.columns] for c in "cluster_id")

            for prop_name in metrics.columns:
                extractor.set_property(key=prop_name, values=metrics[prop_name], ids=metrics["cluster_id"].values)

        return extractor
    
    def get_sorting_hypnogram(
        self, subject: str, experiment: str, alias: str, probe: str, sorting: str, postprocessing: str,
    ):
        main_sorting_dir = self.get_main_sorting_dir(
            subject, experiment, alias, probe, sorting
        )
        postprocessing_dir = main_sorting_dir / postprocessing
        hyp_path = postprocessing_dir / "hypnogram.htsv"
        assert postprocessing_dir.is_dir(), f"Expected Kilosort directory not found: {postprocessing_dir}"

        if not hyp_path.exists():
            import warnings
            warnings.warn(f"No `hypnogram.htsv` file in postpro dir. Returning None")
            return None

        return ece_utils.read_htsv(hyp_path)

    def get_sharptrack(
        self, subject: str, experiment: str, probe: str
    ) -> sharptrack.SHARPTrack:
        """Load SHARPTrack files.

        experiment_params.json should include a probes->imec0->SHARP-Track->filename.mat field.
        filename.mat is then to be found in the same location as experiment_params.json (i.e.e the experiment-subject directory)
        """
        opts = self.load_experiment_subject_json(
            experiment, subject, constants.EXP_PARAMS_FNAME
        )
        fname = opts["probes"][probe]["SHARP-Track"]
        file = self.get_experiment_subject_file(experiment, subject, fname)
        return sharptrack.SHARPTrack(file)
    
    def load_segments_table(
        self,
        wneSubject,
        experiment,
        alias,
        probe,
        sorting,
        return_all_segment_types=False,
    ):
        """Load a sorting's segment file.
        
        Add a couple useful columns: `nSegmentSamp`, `segmentDuration`, `segmentExpmtPrbAcqFirstTime`, `segmentExpmtPrbAcqLastTime`
        """
        segment_file = (
            self.get_alias_subject_directory(experiment, alias, wneSubject.name)
            / f"{sorting}.{probe}"
            / "segments.htsv"
        )
        if not segment_file.exists():
            raise FileNotFoundError(
                f"Segment table not found at {segment_file}."
            )

        segments = ece_utils.read_htsv(
            segment_file
        )

        segments["nSegmentSamp"] = segments["end_frame"] - segments["start_frame"]
        segments["segmentDuration"] = segments["nSegmentSamp"].div(segments["imSampRate"])
        segments["segmentExpmtPrbAcqFirstTime"] = segments["expmtPrbAcqFirstTime"] + \
            segments["start_frame"].div(segments["imSampRate"])
        segments["segmentExpmtPrbAcqLastTime"] = segments["segmentExpmtPrbAcqFirstTime"] + \
            segments["segmentDuration"]

        if return_all_segment_types:
            return segments

        return segments[segments["type"] == "keep"]

    def get_sample2time(
        self,
        wneSubject,
        experiment: str,
        alias: str,
        probe: str,
        sorting: str,
        allow_no_sync_file=False,
    ) -> Callable:

        # TODO: Once permissions are fixed, should come from f = wneProject.get_experiment_subject_file(experiment, wneSubject.name, f"prb_sync.ap.htsv")
        # Load probe sync table.
        probe_sync_file = (
            self.get_alias_subject_directory(experiment, alias, wneSubject.name)
            / f"{sorting}.{probe}"
            / "prb_sync.ap.htsv"
        )
        if not probe_sync_file.exists():
            if allow_no_sync_file:
                logger.info(
                    f"Could not find sync table at {probe_sync_file}.\n"
                    f"`allow_no_sync_file` == True : Ignoring probe sync in sample2time"
                )
                sync_table = None
            else:
                logger.info(f"Could not find sync table at {probe_sync_file}.")
                return None
        else:
            sync_table = ece_utils.read_htsv(
                probe_sync_file
            )  # Used to map this probe's times to imec0.

        # Load segment table
        segments = self.load_segments_table(
            wneSubject,
            experiment,
            alias,
            probe,
            sorting,
            return_all_segment_types=False,
        )  # Used to map SI sorting samples to this probe's times.

        # Get all the good segments (aka the ones in the sorting), in chronological order.
        # Compute which samples in the recording belong to each segment.
        sorted_segments = segments[segments["type"] == "keep"].copy()
        sorted_segments["nSegmentSamples"] = (
            sorted_segments["end_frame"] - sorted_segments["start_frame"]
        )  # N of sorted samples in each segment

        cum_sorted_samples_by_end = sorted_segments[
            "nSegmentSamples"
        ].cumsum()  # N of sorted samples by the end of each segment
        cum_sorted_samples_by_start = cum_sorted_samples_by_end.shift(
            1, fill_value=0
        )  # N of sorted samples by the start of each segment
        sorted_segments[
            "start_sample"
        ] = cum_sorted_samples_by_start  # First sample index of concatenated recording belonging to each semgent
        sorted_segments["end_sample"] = cum_sorted_samples_by_end

        # Given a sample number in the SI recording, we can now figure out:
        #   (1) the segment it came from
        #   (2) the file that segment belongs to
        #   (3) how to map that file's times into our canonical timebase.
        # We make a function that does this for an arbitrary array of sample numbers in the SI object, so we can use it later as needed.
        if sync_table is not None:
            sync_table = sync_table.set_index("source")

        # TODO: Rename start_sample -> si_start_sample?
        def sample2time(s):
            s = s.astype("float")
            for seg in sorted_segments.itertuples():
                mask = (s >= seg.start_sample) & (
                    s < seg.end_sample
                )  # Mask samples belonging to this segment
                s[mask] = (
                    (s[mask] - seg.start_sample) / seg.imSampRate
                    + seg.expmtPrbAcqFirstTime
                    + seg.start_frame / seg.imSampRate
                )  # Convert to number of seconds in this probe's (expmtPrbAcq) timebase
                if sync_table is not None:
                    sync_entry = sync_table.loc[
                        seg.fname
                    ]  # Get info needed to sync to imec0's (expmtPrbAcq) timebase
                    s[mask] = (
                        sync_entry.slope * s[mask] + sync_entry.intercept
                    )  # Sync to imec0 (expmtPrbAcq) timebase
            return s

        return sample2time

    # TODO: Likely deprecated. Remove.
    def remap_probe_times(
        self,
        subject: str,
        experiment: str,
        fromProbe: str,
        times: np.ndarray,
        toProbe: str = "imec0",
    ) -> np.ndarray:
        """Remap a vector of probe times to the canonical experiment timebase, using a subject's precomputed sync models.

        Parameters:
        ===========
        subject: str
        experiment: str
        fromProbe: str
            The probe whose times need remapping.
        times: (nTimes,)
            A vector of times to renamp
        toProbe: str
            The probe whose timebase defines the experiment's canonical time. Always imec0.

        Returns:
        ========
        newTimes: (nTimes,)
        """
        if fromProbe == toProbe:
            return times

        sync_models_file = self.get_experiment_subject_file(
            experiment, subject, "sync_models.pickle"
        )
        assert sync_models_file.is_file(), "Sync models file not found."
        with open(sync_models_file, "rb") as f:
            sync_models = pickle.load(f)

        assert "prb2prb" in sync_models, "Probe-to-probe sync models not found in file."
        model = sync_models["prb2prb"][fromProbe][toProbe]
        return sync.remap_times(times, model)

    def load_hypnogram(
        self, experiment: str, subject: str, simplify: bool = True
    ) -> hypnogram.Hypnogram:
        f = self.get_experiment_subject_file(experiment, subject, constants.HYPNOGRAM_FNAME)
        hyp = hypnogram.FloatHypnogram.from_htsv(f)
        if simplify:
            hyp = hyp.replace_states(constants.SIMPLIFIED_STATES)

        return hyp


class ProjectLibrary:
    def __init__(self, projects_file: Pathlike):
        self.file = Path(projects_file)
        self.yaml_stream = load_yaml_stream(self.file)

    def get_project_document(self, project_name: str) -> dict:
        """Get a project's YAML document from a YAML stream.

        YAML documents must contain a 'project' field:
        ---
        project: project_name
        ...
        """
        matches = [doc for doc in self.yaml_stream if doc["project"] == project_name]
        assert len(matches) == 1, f"Exactly 1 YAML document should match {project_name}"
        return matches[0]

    def get_project(self, project_name: str) -> Project:
        doc = self.get_project_document(project_name)
        return Project(project_name, Path(doc["project_directory"]))
