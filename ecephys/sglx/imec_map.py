import numpy as np
import pandas as pd
from io import StringIO
from .external.SGLXMetaToCoords import XYCoord10
from .external.readSGLX import readMeta
from ..utils import all_equal
from pathlib import PurePath, Path

SUBPACKAGE_DIRECTORY = Path(__file__).resolve().parent


def check_library(fname):
    """Check if a map file is included in the standard library and, if so,
    return the absolute path to it."""
    file = SUBPACKAGE_DIRECTORY / "external" / "maps" / fname
    assert file.is_file(), f"Requested file {fname} is not in the library."
    return file


def parse_imroTbl(imroTbl_string):
    """Parse and check an IMRO table string to a DataFrame.
    The IMRO table string appears in both .imro files and in the ~imroTbl field
    of SGLX .meta files.  See SGLX documentation for details.

    Parameters
    ==========
    imroTbl_string: string
        The string to parse, without newline characters.
    """
    entries = imroTbl_string.strip("()").split(")(")
    header_entry = entries[0]
    channel_entries = entries[1:]

    probe_type, n_chans = (int(s) for s in header_entry.split(","))
    assert probe_type == 0, "Only Neuropixel 1.0 probes are supported."
    imro = pd.read_csv(
        StringIO("\n".join(channel_entries)),
        delim_whitespace=True,
        names=["chan_id", "bank", "ref_id", "ap_gain", "lf_gain", "ap_highpass"],
    )
    imro["ele"] = imro.bank.values * n_chans + imro.chan_id.values
    imro["x"], imro["y"] = XYCoord10({}, imro.ele.values, False)
    return imro


def parse_snsChanMap(snsChanMap_string, assert_stream_type=False):
    """Parse and check an snsChanMap string to a DataFrame.
    This string only appears in the ~snsChanMap field of SGLX .meta files.
    The channel map string in .imec.cmp files has a different format,
    and should be parsed using `read_cmp_file`. See SGLX documentation for details.

    Parameters
    ==========
    snsChanMap_string: string
        The string to parse, without newline characters.
    """
    entries = snsChanMap_string.strip("()").split(")(")
    header_entry = entries[0]
    channel_entries = entries[1:]

    nAP, nLFP, nSY = (int(s) for s in header_entry.split(","))

    cmp = pd.read_csv(
        StringIO("\n".join(channel_entries)),
        sep=";|:",
        names=["label", "acq_order", "usr_order"],
        engine="python",
    )
    cmp["stream"] = cmp["label"].str.extract("(\D+)")
    cmp["chan_id"] = cmp["label"].str.extract("(\d+)").astype(int)
    if assert_stream_type == "LF":
        assert len(cmp) == (nLFP + nSY), "File header does not match content."
    if assert_stream_type == "AP":
        assert len(cmp) == (nAP + nSY), "File header does not match content."

    return cmp


def read_imro_file(file):
    """Parse, check, and read an IMRO file to a DataFrame.

    Parameters
    ===========
    file: string or Path
        The file to read, as an absolute path.
    """
    assert str(file).endswith(".imro"), "Unexpected filename."
    with open(file, "r") as f:
        contents = f.readline().rstrip()

    return parse_imroTbl(contents)


def read_cmp_file(file):
    """Parse, check, and read a channel map file to a DataFrame.
    See SGLX documentation for explanations of DataFrame columns.

    Parameters
    ===========
    file: string or Path
        The file to read, as an absolute path.
    """
    assert str(file).endswith(".imec.cmp"), "Unexpected filename."
    with open(file, "r") as f:
        firstline = f.readline().rstrip()
    nAP, nLFP, nSY = (int(s) for s in firstline.split(","))
    cmp = pd.read_csv(
        file,
        sep=" |;",
        names=["label", "acq_order", "usr_order"],
        skiprows=1,
        engine="python",
    )
    cmp["stream"] = cmp["label"].str.extract("(\D+)")
    cmp["chan_id"] = cmp["label"].str.extract("(\d+)").astype(int)
    assert len(cmp) == (nAP + nLFP + nSY), "File header does not match content."
    return cmp


class ImecMap:
    """Parses and represents SpikeGLX's map files (i.e. the "IMRO table" and "Channel map") as pandas DataFrames.
    Provides methods for exploring, combining, and computing with these files.

    See: https://billkarsh.github.io/SpikeGLX/#interesting-map-files
    See: https://billkarsh.github.io/SpikeGLX/help/imroTables/
    """

    def __init__(self, imro, cmp, map_name="Unknown ImecMap"):
        """Create an ImecMap object.

        Paramters
        =========
        imro: DataFrame
            The IMRO table, as returned by `parse_imroTbl()`.
        cmp: DataFrame
            The channel map table, as returned by `parse_snsChanMap` or `read_cmp_file`.
        map_name: string (optional)
            The name of the map, simply for convenience and bookeeping.
        """
        self._imro = imro
        self._cmp = cmp
        self._map_name = map_name

    def __repr__(self):
        return self._map_name

    @property
    def imro(self):
        """Get just the IMRO table."""
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
    def full(self):
        """Get a combined representaion of both the IMRO and channel map."""
        return self.imro.merge(self.neural_cmp).sort_values("usr_order")

    @property
    def lf_map(self):
        """Get just the LFP portion of the map."""
        return self.full[self.full["stream"] == "LF"]

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

    @classmethod
    def from_library(cls, map_name):
        """Create an ImecMap object from the standard library.
        Example: `m = ImecMap.from_library("LongCol_1Shank")`
        """
        imro = read_imro_file(check_library(map_name + ".imro"))
        cmp = read_cmp_file(check_library(map_name + ".imec.cmp"))
        return cls(imro, cmp, map_name)

    @classmethod
    def from_meta(cls, bin_file):
        """Create an ImecMap object from the metadata associated with a binary file.
        file = '/path/to/my-run_g0_t0.imec0.lf.bin'
        Example: `m = ImecMap.from_meta(file)`
        """
        meta = readMeta(bin_file)
        imro = parse_imroTbl(meta["imroTbl"])
        cmp = parse_snsChanMap(meta["snsChanMap"])
        map_name = PurePath(meta["imRoFile"].stem)
        return cls(imro, cmp, map_name)

    @classmethod
    def LongCol(cls):
        return cls.from_library("LongCol_1shank")

    @classmethod
    def CheckPat(cls):
        return cls.from_library("CheckPat_1shank")

    @classmethod
    def Default(cls):
        return cls.from_library("Default_1shank")

    @classmethod
    def Tetrode(cls):
        return cls.from_library("Tetrode_1shank")
