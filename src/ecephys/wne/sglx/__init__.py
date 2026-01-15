# Only import lightweight modules at package level.
# Heavy modules must be imported directly to avoid slow import times:
#   from ecephys.wne.sglx import spikeinterface
#   from ecephys.wne.sglx import legacy_sorting
#   from ecephys.wne.sglx import pipeline
#   from ecephys.wne.sglx import utils
from . import experiments, sessions
from .project import SGLXProject, SGLXProjectLibrary
from .subject import SGLXSubject, SGLXSubjectLibrary

__all__ = [
    "experiments",
    "sessions",
    "SGLXProject",
    "SGLXProjectLibrary",
    "SGLXSubject",
    "SGLXSubjectLibrary",
]
