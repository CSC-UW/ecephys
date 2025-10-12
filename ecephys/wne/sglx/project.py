import pathlib
from pathlib import Path
from typing import Union

from ecephys.wne.project import Project, ProjectLibrary

Pathlike = Union[Path, str]


class SGLXProject(Project):
    def __init__(self, project_name: str, project_dir: pathlib.Path):
        Project.__init__(self, project_name, project_dir)

    def __repr__(self):
        return f"sglx_project: {self.name}, {self.dir}"


class SGLXProjectLibrary(ProjectLibrary):
    def __init__(self, projects_file: Pathlike):
        ProjectLibrary.__init__(self, projects_file)

    def get_project(self, project_name: str) -> SGLXProject:
        doc = self.get_project_document(project_name)
        return SGLXProject(project_name, Path(doc["project_directory"]))
