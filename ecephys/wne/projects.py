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

# Tom's notes:
# Experiments + aliases assumed, sessions not?
# Only projects.yaml is required to resolve paths?
# You could name a project the same thing as an experiment
# You could name a project "Common" or "Scoring" or "Sorting"
import json
import yaml
import logging
import spikeinterface.extractors as se
from pathlib import Path
from ecephys import utils, sglx, wne, sharptrack

logger = logging.getLogger(__name__)


def load_yaml_stream(yaml_path):
    """Load all YAML documents in a file."""
    with open(yaml_path) as fp:
        yaml_stream = list(yaml.safe_load_all(fp))
    return yaml_stream


class ProjectLibrary:
    def __init__(self, projects_file):
        self.file = Path(projects_file)
        self.yaml_stream = load_yaml_stream(self.file)

    def get_project_document(self, project_name):
        """Get a project's YAML document from a YAML stream.

        YAML documents must contain a 'project' field:
        ---
        project: project_name
        ...
        """
        matches = [doc for doc in self.yaml_stream if doc["project"] == project_name]
        assert len(matches) == 1, f"Exactly 1 YAML document should match {project_name}"
        return matches[0]

    def get_project(self, project_name):
        doc = self.get_project_document(project_name)
        return Project(project_name, Path(doc["project_directory"]))


class Project:
    def __init__(self, project_name, project_dir):
        self.name = project_name
        self.dir = Path(project_dir)

    def __repr__(self):
        return f"{self.name}: {self.dir}"

    #####
    # Methods for getting directories
    #####

    def get_subject_directory(self, subject_name):
        """Get a subject's directory for this project."""
        return self.dir / subject_name

    def get_experiment_directory(self, experiment_name):
        return self.dir / experiment_name

    # TODO: Can we make a separate `get_subalias_directory` function?
    def get_alias_directory(self, experiment_name, alias_name, subalias_idx=None):
        if (subalias_idx is None) or (subalias_idx == -1):
            return self.get_experiment_directory(experiment_name) / alias_name
        else:
            return (
                self.get_experiment_directory(experiment_name)
                / f"{alias_name}_{subalias_idx}"
            )

    def get_experiment_subject_directory(self, experiment_name, subject_name):
        return self.get_experiment_directory(experiment_name) / subject_name

    def get_alias_subject_directory(self, experiment_name, alias_name, subject_name):
        return self.get_alias_directory(experiment_name, alias_name) / subject_name

    def get_alias_subject_directory(
        self, experiment_name, alias_name, subject_name, subalias_idx=None
    ):
        return (
            self.get_alias_directory(
                experiment_name, alias_name, subalias_idx=subalias_idx
            )
            / subject_name
        )

    #####
    # Methods for getting files
    #####

    def get_project_file(self, fname):
        return self.dir / fname

    def get_project_subject_file(self, subject_name, fname):
        return self.get_subject_directory(subject_name) / fname

    def get_experiment_file(self, experiment_name, fname):
        return self.get_experiment_directory(experiment_name) / fname

    def get_alias_file(self, experiment_name, alias_name, fname, subalias_idx=None):
        return (
            self.get_alias_directory(
                experiment_name, alias_name, subalias_idx=subalias_idx
            )
            / fname
        )

    def get_experiment_subject_file(self, experiment_name, subject_name, fname):
        return (
            self.get_experiment_subject_directory(experiment_name, subject_name) / fname
        )

    def get_alias_subject_file(self, experiment_name, alias_name, subject_name, fname):
        return (
            self.get_alias_subject_directory(experiment_name, alias_name, subject_name)
            / fname
        )

    def get_sglx_counterparts(
        self,
        subject_name,
        paths,
        extension,
        remove_probe=False,
        remove_stream=False,
    ):
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
        counterparts = wne.sglx.sessions.mirror_raw_data_paths(
            self.get_subject_directory(subject_name), paths
        )  # Mirror paths at the project's subject directory
        counterparts = [
            sglx.file_mgmt.replace_ftype(p, extension, remove_probe, remove_stream)
            for p in counterparts
        ]
        return utils.remove_duplicates(counterparts)

    def load_experiment_subject_json(self, experiment_name, subject_name, fname):
        path = self.get_experiment_subject_file(experiment_name, subject_name, fname)
        with open(path) as f:
            return json.load(f)

    def get_all_probes(self, subject_name, experiment_name):
        opts = self.load_experiment_subject_json(
            experiment_name, subject_name, wne.constants.EXP_PARAMS_FNAME
        )
        return opts["probes"].keys()

    # TODO: The probe and sortingName arguments could be replaced by a single probeDir argument.
    def get_kilosort_extractor(self, subject, experiment, alias, probe, sortingName):
        """Load the contents of a Kilosort output directory.

        Parameters:
        ===========
        wneProj: wne.Project
            The Project where the sorting output is to be loaded from.
        subject: str
        experiment: str
        alias: str
        probe: str
        sortingName:
            e.g. "ks2_5_catgt_df_postpro_2_metrics_all_isi"

        Returns:
        ========
        KilosortSortingExtractor
        """
        dir = (
            self.get_alias_subject_directory(experiment, alias, subject)
            / f"{sortingName}.{probe}"
        )
        assert dir.is_dir(), f"Expected Kilosort directory not found: {dir}"
        extractor = se.KiloSortSortingExtractor(dir)

        # Check that quality metrics are available
        # TODO: Figure out how to automatically regenerate cluster_info.tsv without making all of phy a dependency, so we can stop doing this manually.
        if not any("isi_viol" in prop for prop in extractor.get_property_keys()):
            msg = "Quality metrics not found. Either they have not been run, or cluster_info.tsv has not been regenerated to include metrics.csv. You can do this manually by opening the data in Phy and clicking 'save'."
            logger.warn(msg)

        return extractor

    def get_sharptrack(self, subject, experiment, probe):
        """Load SHARPTrack files.

        experiment_params.json should include a probes->imec0->SHARP-Track->filename.mat field.
        filename.mat is then to be found in the same location as experiment_params.json (i.e.e the experiment-subject directory)
        """
        opts = self.load_experiment_subject_json(
            experiment, subject, wne.constants.EXP_PARAMS_FNAME
        )
        fname = opts["probes"][probe]["SHARP-Track"]
        file = self.get_experiment_subject_file(experiment, subject, fname)
        return sharptrack.SHARPTrack(file)
