from pathlib import Path

from . import core

SUBPACKAGE_DIRECTORY = Path(__file__).resolve().parent
DATA_DIRECTORY = SUBPACKAGE_DIRECTORY / "data"


def float_hypnogram():
    p = DATA_DIRECTORY / "visbrain_hypnogram.txt"
    return core.FloatHypnogram.from_visbrain(p)


def datetime_hypnogram():
    p = DATA_DIRECTORY / "datetime_hypnogram.tsv"
    return core.DatetimeHypnogram.from_htsv(p)
