# Only import lightweight submodules at package level.
# Heavy processing functions from core.py and plotting functions from plt.py
# must be imported directly to avoid slow import times:
#   from ecephys.xrsig import core
#   from ecephys.xrsig.core import kernel_current_source_density
#   from ecephys.xrsig import plt
from . import senzai, si_extractor

__all__ = [
    "senzai",
    "si_extractor",
]


# Lazy imports for expensive visualization modules
_LAZY_IMPORTS = {
    "ephyviewer": ".ephyviewer",
}


def __getattr__(name):
    """Lazy import expensive modules only when accessed."""
    if name in _LAZY_IMPORTS:
        import importlib

        module = importlib.import_module(_LAZY_IMPORTS[name], package=__package__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
