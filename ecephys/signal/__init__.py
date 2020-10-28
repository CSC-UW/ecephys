import numpy as np


def rms(data):
    """
    Computes root-mean-squared voltage of flattened array

    Input:
    -----
    data - numpy.ndarray

    Output:
    ------
    rms_value - float
    """

    return np.power(
        np.mean(
            np.power(data.astype('float32'), 2)
        ),
        0.5
    )
