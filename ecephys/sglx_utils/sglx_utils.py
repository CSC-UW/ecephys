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


# ShankMap for Neuropixel 1.0 probes
snsShankMap = "(1,2,480)(0:0:0:1)(0:1:0:1)(0:0:1:1)(0:1:1:1)(0:0:2:1)(0:1:2:1)(0:0:3:1)(0:1:3:1)(0:0:4:1)(0:1:4:1)(0:0:5:1)(0:1:5:1)(0:0:6:1)(0:1:6:1)(0:0:7:1)(0:1:7:1)(0:0:8:1)(0:1:8:1)(0:0:9:1)(0:1:9:1)(0:0:10:1)(0:1:10:1)(0:0:11:1)(0:1:11:1)(0:0:12:1)(0:1:12:1)(0:0:13:1)(0:1:13:1)(0:0:14:1)(0:1:14:1)(0:0:15:1)(0:1:15:1)(0:0:16:1)(0:1:16:1)(0:0:17:1)(0:1:17:1)(0:0:18:1)(0:1:18:1)(0:0:19:1)(0:1:19:1)(0:0:20:1)(0:1:20:1)(0:0:21:1)(0:1:21:1)(0:0:22:1)(0:1:22:1)(0:0:23:1)(0:1:23:1)(0:0:24:1)(0:1:24:1)(0:0:25:1)(0:1:25:1)(0:0:26:1)(0:1:26:1)(0:0:27:1)(0:1:27:1)(0:0:28:1)(0:1:28:1)(0:0:29:1)(0:1:29:1)(0:0:30:1)(0:1:30:1)(0:0:31:1)(0:1:31:1)(0:0:32:1)(0:1:32:1)(0:0:33:1)(0:1:33:1)(0:0:34:1)(0:1:34:1)(0:0:35:1)(0:1:35:1)(0:0:36:1)(0:1:36:1)(0:0:37:1)(0:1:37:1)(0:0:38:1)(0:1:38:1)(0:0:39:1)(0:1:39:1)(0:0:40:1)(0:1:40:1)(0:0:41:1)(0:1:41:1)(0:0:42:1)(0:1:42:1)(0:0:43:1)(0:1:43:1)(0:0:44:1)(0:1:44:1)(0:0:45:1)(0:1:45:1)(0:0:46:1)(0:1:46:1)(0:0:47:1)(0:1:47:1)(0:0:48:1)(0:1:48:1)(0:0:49:1)(0:1:49:1)(0:0:50:1)(0:1:50:1)(0:0:51:1)(0:1:51:1)(0:0:52:1)(0:1:52:1)(0:0:53:1)(0:1:53:1)(0:0:54:1)(0:1:54:1)(0:0:55:1)(0:1:55:1)(0:0:56:1)(0:1:56:1)(0:0:57:1)(0:1:57:1)(0:0:58:1)(0:1:58:1)(0:0:59:1)(0:1:59:1)(0:0:60:1)(0:1:60:1)(0:0:61:1)(0:1:61:1)(0:0:62:1)(0:1:62:1)(0:0:63:1)(0:1:63:1)(0:0:64:1)(0:1:64:1)(0:0:65:1)(0:1:65:1)(0:0:66:1)(0:1:66:1)(0:0:67:1)(0:1:67:1)(0:0:68:1)(0:1:68:1)(0:0:69:1)(0:1:69:1)(0:0:70:1)(0:1:70:1)(0:0:71:1)(0:1:71:1)(0:0:72:1)(0:1:72:1)(0:0:73:1)(0:1:73:1)(0:0:74:1)(0:1:74:1)(0:0:75:1)(0:1:75:1)(0:0:76:1)(0:1:76:1)(0:0:77:1)(0:1:77:1)(0:0:78:1)(0:1:78:1)(0:0:79:1)(0:1:79:1)(0:0:80:1)(0:1:80:1)(0:0:81:1)(0:1:81:1)(0:0:82:1)(0:1:82:1)(0:0:83:1)(0:1:83:1)(0:0:84:1)(0:1:84:1)(0:0:85:1)(0:1:85:1)(0:0:86:1)(0:1:86:1)(0:0:87:1)(0:1:87:1)(0:0:88:1)(0:1:88:1)(0:0:89:1)(0:1:89:1)(0:0:90:1)(0:1:90:1)(0:0:91:1)(0:1:91:1)(0:0:92:1)(0:1:92:1)(0:0:93:1)(0:1:93:1)(0:0:94:1)(0:1:94:1)(0:0:95:1)(0:1:95:0)(0:0:96:1)(0:1:96:1)(0:0:97:1)(0:1:97:1)(0:0:98:1)(0:1:98:1)(0:0:99:1)(0:1:99:1)(0:0:100:1)(0:1:100:1)(0:0:101:1)(0:1:101:1)(0:0:102:1)(0:1:102:1)(0:0:103:1)(0:1:103:1)(0:0:104:1)(0:1:104:1)(0:0:105:1)(0:1:105:1)(0:0:106:1)(0:1:106:1)(0:0:107:1)(0:1:107:1)(0:0:108:1)(0:1:108:1)(0:0:109:1)(0:1:109:1)(0:0:110:1)(0:1:110:1)(0:0:111:1)(0:1:111:1)(0:0:112:1)(0:1:112:1)(0:0:113:1)(0:1:113:1)(0:0:114:1)(0:1:114:1)(0:0:115:1)(0:1:115:1)(0:0:116:1)(0:1:116:1)(0:0:117:1)(0:1:117:1)(0:0:118:1)(0:1:118:1)(0:0:119:1)(0:1:119:1)(0:0:120:1)(0:1:120:1)(0:0:121:1)(0:1:121:1)(0:0:122:1)(0:1:122:1)(0:0:123:1)(0:1:123:1)(0:0:124:1)(0:1:124:1)(0:0:125:1)(0:1:125:1)(0:0:126:1)(0:1:126:1)(0:0:127:1)(0:1:127:1)(0:0:128:1)(0:1:128:1)(0:0:129:1)(0:1:129:1)(0:0:130:1)(0:1:130:1)(0:0:131:1)(0:1:131:1)(0:0:132:1)(0:1:132:1)(0:0:133:1)(0:1:133:1)(0:0:134:1)(0:1:134:1)(0:0:135:1)(0:1:135:1)(0:0:136:1)(0:1:136:1)(0:0:137:1)(0:1:137:1)(0:0:138:1)(0:1:138:1)(0:0:139:1)(0:1:139:1)(0:0:140:1)(0:1:140:1)(0:0:141:1)(0:1:141:1)(0:0:142:1)(0:1:142:1)(0:0:143:1)(0:1:143:1)(0:0:144:1)(0:1:144:1)(0:0:145:1)(0:1:145:1)(0:0:146:1)(0:1:146:1)(0:0:147:1)(0:1:147:1)(0:0:148:1)(0:1:148:1)(0:0:149:1)(0:1:149:1)(0:0:150:1)(0:1:150:1)(0:0:151:1)(0:1:151:1)(0:0:152:1)(0:1:152:1)(0:0:153:1)(0:1:153:1)(0:0:154:1)(0:1:154:1)(0:0:155:1)(0:1:155:1)(0:0:156:1)(0:1:156:1)(0:0:157:1)(0:1:157:1)(0:0:158:1)(0:1:158:1)(0:0:159:1)(0:1:159:1)(0:0:160:1)(0:1:160:1)(0:0:161:1)(0:1:161:1)(0:0:162:1)(0:1:162:1)(0:0:163:1)(0:1:163:1)(0:0:164:1)(0:1:164:1)(0:0:165:1)(0:1:165:1)(0:0:166:1)(0:1:166:1)(0:0:167:1)(0:1:167:1)(0:0:168:1)(0:1:168:1)(0:0:169:1)(0:1:169:1)(0:0:170:1)(0:1:170:1)(0:0:171:1)(0:1:171:1)(0:0:172:1)(0:1:172:1)(0:0:173:1)(0:1:173:1)(0:0:174:1)(0:1:174:1)(0:0:175:1)(0:1:175:1)(0:0:176:1)(0:1:176:1)(0:0:177:1)(0:1:177:1)(0:0:178:1)(0:1:178:1)(0:0:179:1)(0:1:179:1)(0:0:180:1)(0:1:180:1)(0:0:181:1)(0:1:181:1)(0:0:182:1)(0:1:182:1)(0:0:183:1)(0:1:183:1)(0:0:184:1)(0:1:184:1)(0:0:185:1)(0:1:185:1)(0:0:186:1)(0:1:186:1)(0:0:187:1)(0:1:187:1)(0:0:188:1)(0:1:188:1)(0:0:189:1)(0:1:189:1)(0:0:190:1)(0:1:190:1)(0:0:191:1)(0:1:191:1)"
