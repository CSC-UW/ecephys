import sglxarray
import numpy as np
from pathlib import Path

data_path = Path(sglxarray.__path__[0]) / "data"


def example_data_path():
    return data_path / "example-data.exported.imec0.lf.bin"


def example_data():
    p = example_data_path()
    chans = np.arange(0, 384)

    return sglxarray.load_trigger(p, chans)
