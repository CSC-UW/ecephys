import numpy as np


def get_user_order():
    return np.concatenate((np.arange(0, 384, 2), np.arange(1, 384, 2)))