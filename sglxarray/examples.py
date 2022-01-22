import sglxarray
import numpy as np
from pathlib import Path

SUBPACKAGE_DIRECTORY = Path(__file__).resolve().parent
DATA_DIRECTORY = SUBPACKAGE_DIRECTORY / "data"


def example_data_path():
    return DATA_DIRECTORY / "example-data.exported.imec0.lf.bin"


def example_data():
    p = example_data_path()
    chans = np.arange(0, 384)

    return sglxarray.load_trigger(p, chans)
