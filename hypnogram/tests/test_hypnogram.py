import os.path as op
import numpy.testing as npt
import hypnogram

data_path = op.join(hypnogram.__path__[0], "data")
# Load data like: op.join(data_path, 'mydatafile.dat')


def test_trivial():
    """
    Should always pass. Just used to ensure that py.test is setup correctly.
    """
    npt.assert_equal(np.array([1, 1, 1]), np.array([1, 1, 1]))