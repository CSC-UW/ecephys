import os.path as op
import sglxarray
import numpy as np

data_path = op.join(sglxarray.__path__[0], "data")
# Load data like: op.join(data_path, 'mydatafile.dat')


def example_data(chans=None):
    p = op.join(data_path, "example-data.exported.imec0.lf.bin")
    chans = chans or np.arange(0, 384, 364)

    return sglxarray.load_trigger(p, chans)
