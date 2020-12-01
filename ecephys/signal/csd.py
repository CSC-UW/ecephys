import numpy as np
from kcsd import KCSD1D


def get_kcsd(lfps, intersite_distance=0.020, do_lcurve=False, **kwargs):
    n_chans = lfps.shape[1]
    ele_pos = np.linspace(0.0, (n_chans - 1) * intersite_distance, n_chans).reshape(
        n_chans, 1
    )

    k = KCSD1D(ele_pos, lfps.T, **kwargs)

    if do_lcurve:
        print("Performing L-curve parameter estimation...")
        k.L_curve()

    return k