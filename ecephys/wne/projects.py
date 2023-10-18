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
from typing import Optional, Union

import numpy as np
import pandas as pd
import yaml

import ecephys.sglx
import ecephys.utils
import spikeinterface.extractors as se
from ecephys import hypnogram, sharptrack, sync
from ecephys.wne import constants

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

    def get_project_directory(self) -> Path:
        return self.dir

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

    def load_experiment_subject_params(self, experiment: str, subject: str) -> dict:
        return self.load_experiment_subject_json(
            experiment, subject, constants.EXP_PARAMS_FNAME
        )

    def get_all_probes(self, subject: str, experiment: str) -> list[str]:
        opts = self.load_experiment_subject_json(
            experiment, subject, constants.EXP_PARAMS_FNAME
        )
        return list(opts["probes"].keys())

    def get_kilosort_extractor(
        self,
        subject: str,
        experiment: str,
        probe: str,
        alias: str = "full",
        sorting: str = "sorting",
        postprocessing: str = "postpro",
    ) -> se.KiloSortSortingExtractor:
        """Load the contents of a Kilosort output directory. This takes ~20-25s per 100 clusters.

        We keep only a subset of the properties loaded by si.read_kilosort(), since some of those
        may contain obsolete metrics, added to the sorting directory and integrated into cluster_info.tsv
        to inform curation.

        We then add as properties all the metrics from the postprocessing directory (originating from both
        from the "metrics"  and "waveform_metrics" spikeinterface extensions)
        """

        PROPERTIES_FROM_SORTING_DIR = [
            "Amplitude",
            "ContamPct",
            "KSLabel",
            "acronym",
            "amp",
            "ch",
            "depth",
            "fr",
            "n_spikes",
            "quality",
            "sh",
            "structure",
        ]

        main_sorting_dir = (
            self.get_alias_subject_directory(experiment, alias, subject)
            / f"{sorting}.{probe}"
        )
        sorter_output_dir = main_sorting_dir / "si_output/sorter_output"
        assert (
            sorter_output_dir.is_dir()
        ), f"Expected Kilosort directory not found: {sorter_output_dir}"
        assert (
            sorter_output_dir / "spike_times.npy"
        ).exists(), f"Expected `spike_times.npy` file in: {sorter_output_dir}"
        extractor = se.read_kilosort(sorter_output_dir, keep_good_only=False)

        # Keep only properties of interest
        for property in extractor.get_property_keys():
            if property not in PROPERTIES_FROM_SORTING_DIR:
                extractor.delete_property(property)

        # Load metrics and add as properties
        postprocessing_dir = main_sorting_dir / postprocessing
        if not postprocessing_dir.is_dir():
            import warnings

            warnings.warn(
                f"Could not find postprocessing dir. Ignoring metrics: {postprocessing_dir}"
            )
        else:
            
            # "regular" metrics, already aggregated across vigilance states
            metrics_path = postprocessing_dir / "metrics.csv"
            assert (
                metrics_path.exists()
            ), f"Expected `metrics.csv` file in: {postprocessing_dir}"

            metrics = pd.read_csv(metrics_path)
            # Check correct ids 
            assert set(metrics["cluster_id"].values) == set(extractor.get_unit_ids())

            for prop_name in metrics.columns:
                extractor.set_property(
                    key=prop_name,
                    values=metrics[prop_name],
                    ids=metrics["cluster_id"].values,
                )
            
            # "template_metrics"
            template_metrics_path = postprocessing_dir/"si_output/template_metrics/metrics.csv"
            if not template_metrics_path.exists():
                import warnings
                warnings.warn("Could not find `template_metrics.csv` file. Ignoring template metrics.")
            else:
                template_metrics = pd.read_csv(template_metrics_path, index_col=0)
                # Check correct ids 
                assert set(template_metrics.index.values) == set(extractor.get_unit_ids())
                # Check we're not overriding a property
                assert not any([c in extractor.get_property_keys() for c in template_metrics.columns])

                for prop_name in template_metrics.columns:
                    extractor.set_property(
                        key=prop_name,
                        values=template_metrics[prop_name],
                        ids=template_metrics.index.values,
                    )

        return extractor

    def get_sharptrack(
        self, subject: str, experiment: str, probe: str
    ) -> sharptrack.SHARPTrack:
        """Load SHARPTrack files.

        experiment_params.json should include a probes->`probe_name`->SHARP-Track->filename.mat field.
        filename.mat is then to be found in the same location as experiment_params.json (i.e.e the experiment-subject directory)
        """
        opts = self.load_experiment_subject_json(
            experiment, subject, constants.EXP_PARAMS_FNAME
        )
        fname = opts["probes"][probe]["SHARP-Track"]
        file = self.get_experiment_subject_file(experiment, subject, fname)
        return sharptrack.SHARPTrack(file)

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

    def load_float_hypnogram(
        self,
        experiment: str,
        subject: str,
        simplify: bool = True,
    ) -> hypnogram.FloatHypnogram:
        f = self.get_experiment_subject_file(
            experiment, subject, constants.HYPNOGRAM_FNAME
        )
        hg = hypnogram.FloatHypnogram.from_htsv(f)
        if simplify:
            hg = hg.replace_states(constants.SIMPLIFIED_STATES)
        return hg

    def load_offs_df(
        self,
        experiment: str,
        subject: str,
        probe: str,
        off_fname_suffix: str = constants.DF_OFF_FNAME_SUFFIX,
    ):
        """Load and aggregate off files across structures.

        Loads and aggregate all files of the form
        `<probe>.<acronym>.<off_fname_suffix>` in the `offs` subdirectory
        of the project's experiment_subject_directory.
        """
        off_dir = self.get_experiment_subject_directory(experiment, subject) / "offs"

        structure_offs = []
        for f in off_dir.glob(f"{probe}.*{off_fname_suffix}"):
            structure_offs.append(ecephys.utils.read_htsv(f))

        return pd.concat(structure_offs).reset_index(drop=True)


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
