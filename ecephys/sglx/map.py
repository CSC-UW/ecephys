import numpy as np
from .map_io import load_cmp, load_imro
from .external.SGLXMetaToCoords import XYCoord10
from ..utils import all_equal


class Map:
    """Parses and represents SpikeGLX's map files (i.e. the "IMRO table" and "Channel map") as pandas DataFrames.
    Provides methods for exploring, combining, and computing with these files.

    See: https://billkarsh.github.io/SpikeGLX/#interesting-map-files
    See: https://billkarsh.github.io/SpikeGLX/help/imroTables/
    """

    def __init__(self, map_name):
        """Load files associated with the map_name provided.

        For example: `Map('LongCol_1shank')` creates an object representing
        'LongCol_1shank.imec.cmp and 'LongCol_1shank.imro'
        """
        self._map_name = map_name
        self._imro = load_imro(map_name + ".imro")
        self._cmp = load_cmp(map_name + ".imec.cmp")

    def __repr__(self):
        return self._map_name

    @property
    def imro(self):
        """Get just the IMRO table for this map."""
        return self._imro

    @property
    def cmp(self):
        """Get just the channel map."""
        return self._cmp

    @property
    def neural_cmp(self):
        """Get just the portions of the channel map corresponding to neural data."""
        return self.cmp[(self.cmp.stream == "AP") | (self.cmp.stream == "LF")]

    @property
    def map(self):
        """Get a combined representaion of both the IMRO and channel map."""
        return self.imro.merge(self.neural_cmp).sort_values("usr_order")

    @property
    def lf_map(self):
        """Get just the LFP portion of the map."""
        return self.map[self.map["stream"] == "LF"]

    @property
    def lf_chans(self):
        """Get the LFP channel IDs in user order, as displayed in SGLX."""
        return self.lf_map.chan_id.values

    @property
    def chans(self):
        """Get the channel IDs in user order, as displayed in SGLX."""
        return self.lf_chans

    @property
    def x(self):
        """Get the x coordinates of each channel in self.chans, in microns."""
        return self.lf_map.x.values

    @property
    def y(self):
        """Get the y coordinates of each channel in self.chans, in microns."""
        return self.lf_map.y.values

    @property
    def coords(self):
        """Get the x,y coordinates of each channel in self.chans, in microns."""
        return np.dstack((self.x, self.y)).squeeze()

    @property
    def pitch(self):
        """Get the vertical spacing between electrode sites, in microns"""
        vals = np.diff(self.y)
        assert all_equal(vals), "Electrode pitch is not uniform."
        return vals[0]

    def plot_electrodes(self):
        """Plot the locations of all channels and electrodes."""
        XYCoord10({}, self.imro.ele.values, True)

    def chans2coords(self, chans):
        """Convert channel IDs to their coordinates"""
        df = self.lf_map.set_index("chan_id").loc[chans]
        return np.dstack((df.x.values, df.y.values)).squeeze()

    def y2chans(self, y):
        """Convert y coordinates probe space in microns to the corresponding channel IDs."""
        return self.lf_map.set_index("y").loc[y]


class LongColMap(Map):
    def __init__(self):
        super().__init__("LongCol_1shank")


class CheckPatMap(Map):
    def __init__(self):
        super().__init__("CheckPat_1shank")
