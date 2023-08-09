__version__ = "0.0.1"  # TODO: Use importlib to get this properly from pyproject.toml.
# The version string is needed by Spikeinterface when serializing and deserializing extractor objects from this library during multiprocessing.

from . import (
    sharptrack,
    data_mgmt,
    emg_from_lfp,
    hypnogram,
    plot,
    sglx,
    sglxr,
    npsig,
    tdtxr,
    sync,
    units,
    utils,
    xrsig,
    wne,
)
