import os.path as op

import ecephys.hypnogram as hyp

from . import core

data_path = op.join(hyp.__path__[0], "data")
# Load data like: op.join(data_path, 'mydatafile.dat')


def float_hypnogram():
    p = op.join(data_path, "visbrain_hypnogram.txt")
    return core.load_visbrain_hypnogram(p)


def datetime_hypnogram():
    p = op.join(data_path, "datetime_hypnogram.tsv")
    return core.load_datetime_hypnogram(p)
