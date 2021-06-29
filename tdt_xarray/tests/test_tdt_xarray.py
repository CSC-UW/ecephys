import tdt_xarray as txr
import os.path as op
import numpy as np
import numpy.testing as npt

data_path = op.join(txr.__path__[0], "data")
# Load data like: op.join(data_path, 'mydatafile.dat')


def test_trivial():
    """
    Should always pass. Just used to ensure that py.test is setup correctly.
    """
    npt.assert_equal(np.array([1, 1, 1]), np.array([1, 1, 1]))