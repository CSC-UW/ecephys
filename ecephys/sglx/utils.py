from pathlib import Path

from .external import SGLXMetaToCoords
from .external.readSGLX import (
    readMeta,
    SampRate,
)


def get_meta_path(binpath):
    metaName = binpath.stem + ".meta"
    return Path(binpath).parent / metaName


def get_meta(binpath):
    return readMeta(binpath)


def get_sf(binpath):
    return SampRate(get_meta(binpath))


def get_xy_coords(binpath, **kwargs):
    """Return AP channel indices and their x and y coordinates."""
    metapath = Path(binpath).with_suffix(".meta")
    chans, xcoord, ycoord = SGLXMetaToCoords.MetaToCoords(metapath, 4, **kwargs)
    return chans, xcoord, ycoord