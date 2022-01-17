import numpy as np
from .map_io import load_cmp, load_imro
from .external.SGLXMetaToCoords import XYCoord10


class Map:
    def __init__(self, map_name):
        self._map_name = map_name
        self._imro = load_imro(map_name + ".imro")
        self._cmp = load_cmp(map_name + ".imec.cmp")

    def __repr__(self):
        return self._map_name

    @property
    def imro(self):
        return self._imro

    @property
    def cmp(self):
        return self._cmp

    @property
    def neural_cmp(self):
        return self.cmp[(self.cmp.stream == "AP") | (self.cmp.stream == "LF")]

    @property
    def map(self):
        return self.imro.merge(self.neural_cmp).sort_values("usr_order")

    @property
    def lf_map(self):
        return self.map[self.map["stream"] == "LF"]

    @property
    def lf_chans(self):
        return self.lf_map.chan_id.values

    @property
    def chans(self):
        return self.lf_chans

    @property
    def x(self):
        return self.lf_map.x.values

    @property
    def y(self):
        return self.lf_map.y.values

    @property
    def coords(self):
        return np.dstack((self.x, self.y)).squeeze()

    def plot_electrodes(self):
        XYCoord10({}, self.imro.ele.values, True)

    def chans2coords(self, chans):
        df = self.lf_map.set_index("chan_id").loc[chans]
        return np.dstack((df.x.values, df.y.values)).squeeze()


class LongColMap(Map):
    def __init__(self):
        super().__init__("LongCol_1shank")


class CheckPatMap(Map):
    def __init__(self):
        super().__init__("CheckPat_1shank")
