__version__ = "0.0.1"  # TODO: Use importlib to get this properly from pyproject.toml.
# The version string is needed by Spikeinterface when serializing and deserializing extractor objects from this library during multiprocessing.

from . import (
    emg_from_lfp,
    hypnogram,
    npsig,
    plot,
    sglx,
    sglxr,
    sharptrack,
    sync,
    tdtxr,
    units,
    utils,
    wne,
    xrsig,
)

__all__ = [
    "emg_from_lfp",
    "hypnogram",
    "npsig",
    "plot",
    "sglx",
    "sglxr",
    "sharptrack",
    "sync",
    "tdtxr",
    "units",
    "utils",
    "wne",
    "xrsig",
]
