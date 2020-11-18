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


def median_subtract(sig):
    """Median subtraction on a per-channel basis.

    Parameters
    -----------
    sig : (n_samples,) or (n_samples, n_chans) array
        Time series.

    Returns
    -------
    sig: Same shape as input
        Time series with median subtraction applied.
    """
    return sig - np.median(sig, axis=0)