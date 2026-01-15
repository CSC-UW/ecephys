# Only import lightweight modules at package level.
# Heavy modules must be imported directly to avoid slow import times:
#   from ecephys.wne import sglx
#   from ecephys.wne import siutils
#   from ecephys.wne import utils
from . import constants, project, subject
from .constants import (
    SIMPLIFIED_ARTIFACTS,
    SIMPLIFIED_STATES,
    SYNC_FNAME_MAP,
    Files,
)
from .project import Project, ProjectLibrary
from .subject import Subject

__all__ = [
    "Files",
    "Project",
    "ProjectLibrary",
    "SIMPLIFIED_ARTIFACTS",
    "SIMPLIFIED_STATES",
    "SYNC_FNAME_MAP",
    "constants",
    "project",
    "subject",
    "Subject",
]
