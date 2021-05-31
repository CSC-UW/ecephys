import os.path as op
import hypnogram as hg

data_path = op.join(hg.__path__[0], "data")
# Load data like: op.join(data_path, 'mydatafile.dat')


def float_hypnogram():
    p = op.join(data_path, "3-17-2020_g0_t0.imec0.hypnogram.txt")
    return hg.load_visbrain_hypnogram(p)


def datetime_hypnogram():
    p = op.join(data_path, "3-17-2020_g0_t0.imec0.hypnogram.tsv")
    return hg.load_datetime_hypnogram(p)
