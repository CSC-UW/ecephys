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
from pathlib import Path
from .. import utils
from ..sglx import file_mgmt as sglx_file_mgmt
from .sglx import sessions as sglx_sessions


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
        counterparts = sglx_sessions.mirror_raw_data_paths(
            self.get_subject_directory(subject_name), paths
        )  # Mirror paths at the project's subject directory
        counterparts = [
            sglx_file_mgmt.replace_ftype(p, extension, remove_probe, remove_stream)
            for p in counterparts
        ]
        return utils.remove_duplicates(counterparts)

    def load_experiment_subject_json(self, experiment_name, subject_name, fname):
        path = self.get_experiment_subject_file(experiment_name, subject_name, fname)
        with open(path) as f:
            return json.load(f)
