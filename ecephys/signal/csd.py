import numpy as np
from kcsd import KCSD1D


def kcsd_npix(sig, intersite_distance=0.020, interestimate_distance=0.020):
    n_chans = sig.shape[1]
    ele_pos = np.linspace(0.0, (n_chans - 1) * intersite_distance, n_chans).reshape(
        n_chans, 1
    )
    if interestimate_distance == "default":
        k = KCSD1D(ele_pos, sig.T)
    else:
        k = KCSD1D(ele_pos, sig.T, gdx=interestimate_distance)

    print("doing L-curve")
    k.L_curve()

    return k.values("CSD"), k.estm_x
