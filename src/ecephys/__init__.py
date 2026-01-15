__version__ = "0.0.1"  # TODO: Use importlib to get this properly from pyproject.toml.
# The version string is needed by Spikeinterface when serializing and deserializing extractor objects from this library during multiprocessing.

# Only import lightweight utils module at package level.
# All other submodules must be imported directly to avoid slow import times:
#   from ecephys import hypnogram
#   from ecephys import xrsig
#   from ecephys.xrsig import core as xrsig_core
from . import utils

__all__ = [
    "utils",
]
