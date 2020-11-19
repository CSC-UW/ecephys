import os.path
from pathlib import Path
import numpy as np

from . import SGLXMetaToCoords
from .readSGLX import makeMemMapRaw, readMeta, SampRate, GainCorrectIM


def get_metadata(binpath):
    """Return metadata structure from path to bin."""
    return readSGLX.readMeta(binpath)


def get_srate(binpath):
    """Return sampling rate from path to bin."""
    return readSGLX.SampRate(get_metadata(binpath))


def get_channel_labels(binpath):
    """Return labels parsed from meta['snsChanMap'] for bin's channels."""
    return


def get_channel_map(binpath):
    """Return mapping parsed from meta['snsChanMap'] for bin's channels."""
    return


def get_xy_coords(binpath):
    """Return AP channel indices and their x and y coordinates."""
    metapath = Path(binpath).with_suffix(".meta")
    chans, xcoord, ycoord = SGLXMetaToCoords.MetaToCoords(metapath, 4)
    return chans, xcoord, ycoord


# Functions below are copied from Allen Institute's ecephys_spike_sorting
# package


def EphysParams(ap_band_file):
    # assume metadata file is in same directory as binary, Constuct metadata path

    # read metadata

    metaName, binExt = os.path.splitext(ap_band_file)
    metaFullPath = Path(metaName + ".meta")
    meta = readSGLX.readMeta(metaFullPath)

    if "imDatPrb_type" in meta:
        pType = meta["imDatPrb_type"]
        if pType == "0":
            probe_type = "NP1"
        else:
            probe_type = "NP" + pType
    else:
        probe_type = "3A"  # 3A probe

    sample_rate = float(meta["imSampRate"])

    num_channels = int(meta["nSavedChans"])

    uVPerBit = Chan0_uVPerBit(meta)

    return (probe_type, sample_rate, num_channels, uVPerBit)


# Return gain for imec channels.
# Index into these with the original (acquired) channel IDs.
#
def Chan0_uVPerBit(meta):
    # Returns uVPerBit conversion factor for channel 0
    # If all channels have the same gain (usually set that way for
    # 3A and NP1 probes; always true for NP2 probes), can use
    # this value for all channels.

    imroList = meta["imroTbl"].split(sep=")")
    # One entry for each channel plus header entry,
    # plus a final empty entry following the last ')'
    # channel zero is the 2nd element in the list

    if "imDatPrb_dock" in meta:
        # NP 2.0; APGain = 80 for all channels
        # voltage range = 1V
        # 14 bit ADC
        uVPerBit = (1e6) * (1.0 / 80) / pow(2, 14)
    else:
        # 3A, 3B1, 3B2 (NP 1.0)
        # voltage range = 1.2V
        # 10 bit ADC
        currList = imroList[1].split(sep=" ")  # 2nd element in list, skipping header
        APgain = float(currList[3])
        uVPerBit = (1e6) * (1.2 / APgain) / pow(2, 10)

    return uVPerBit


def load_timeseries(bin_path, chans, start_time=None, end_time=None):
    """Load SpikeGLX timeseries data.

    Parameters
    ----------
    bin_path: joblib Path object
        The path to the binary data (i.e. *.bin)
    chans: 1d array
        The list of channels to load
    start_time: float, optional, default: None
        Start time of the data to load, relative to the file start, in seconds.
        If `None`, load from the start of the file.
    end_time: float, optional, default: None
        End time of the data to load, relative to the file start, in seconds.
        If `None`, load until the end of the file.

    Returns
    -------
    time : 1d array, (n_samples, )
        Time of the data, in seconds from the file start.
    sig: 2d array, (n_samples, n_chans)
        Gain-converted signal
    fs: float
        The sampling frequency of the data
    """

    meta = readMeta(bin_path)
    rawData = makeMemMapRaw(bin_path, meta)
    fs = SampRate(meta)

    # Calculate desire start and end samples
    if start_time:
        firstSamp = int(fs * start_time)
    else:
        firstSamp = 0

    if end_time:
        lastSamp = int(fs * end_time)
    else:
        nFileChan = int(meta["nSavedChans"])
        nFileSamp = int(int(meta["fileSizeBytes"]) / (2 * nFileChan))
        lastSamp = nFileSamp - 1

    # array of times for plot
    time = np.arange(firstSamp, lastSamp + 1)
    time = time / fs  # plot time axis in seconds

    selectData = rawData[chans, firstSamp : lastSamp + 1]
    if meta["typeThis"] == "imec":
        # apply gain correction and convert to uV
        sig = 1e6 * GainCorrectIM(selectData, chans, meta)
    else:
        MN, MA, XA, DW = ChannelCountsNI(meta)
        # print("NI channel counts: %d, %d, %d, %d" % (MN, MA, XA, DW))
        # apply gain coorection and conver to mV
        sig = 1e3 * GainCorrectNI(selectData, chans, meta)

    return time, sig.T, fs
