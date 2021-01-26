import os.path
from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd

from .external import SGLXMetaToCoords
from .external.readSGLX import makeMemMapRaw, readMeta, SampRate, GainCorrectIM


def get_xy_coords(binpath):
    """Return AP channel indices and their x and y coordinates."""
    metapath = Path(binpath).with_suffix(".meta")
    chans, xcoord, ycoord = SGLXMetaToCoords.MetaToCoords(metapath, 4)
    return chans, xcoord, ycoord


def load_timeseries(
    bin_path, chans, start_time=None, end_time=None, datetime=False, xarray=False
):
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
    datetime: boolean
        If `True`, return time as a datetime vector with nanosecond resolution.
        Note that this WILL result in a loss of timing resolution, but it is marginal.
    xarray: boolean
        If `True`, return data as a DataArray

    Returns
    -------
    if xarray=`False`:
        time : 1d array, (n_samples, )
            Time of the data, in seconds from the file start.
        sig: 2d array, (n_samples, n_chans)
            Gain-converted signal
        fs: float
            The sampling frequency of the data
    if xarray=`True`:
        data : xr.DataArray (n_samples, n_chans)
            With labeled dimensions `time` and `channel`, and `fs` attribute.
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

    # Get timestamps of each sample
    time = np.arange(firstSamp, lastSamp + 1)
    time = time / fs  # timestamps in seconds from start of file
    if datetime:
        start_dt = pd.to_datetime(meta["fileCreateTime"]) + pd.to_timedelta(
            time.min(), "s"
        )
        end_dt = start_dt + pd.to_timedelta(time.max(), "s")
        time = pd.date_range(start_dt, end_dt, periods=len(time))

    selectData = rawData[chans, firstSamp : lastSamp + 1]
    if meta["typeThis"] == "imec":
        # apply gain correction and convert to uV
        sig = 1e6 * GainCorrectIM(selectData, chans, meta)
        sig_units = "uV"
    else:
        MN, MA, XA, DW = ChannelCountsNI(meta)
        # print("NI channel counts: %d, %d, %d, %d" % (MN, MA, XA, DW))
        # apply gain coorection and conver to mV
        sig = 1e3 * GainCorrectNI(selectData, chans, meta)
        sig_units = "mV"

    if xarray:
        data = xr.DataArray(
            sig.T, dims=("time", "channel"), coords={"time": time, "channel": chans}
        )
        data.attrs["units"] = sig_units
        data.attrs["fs"] = fs
    else:
        data = (time, sig.T, fs)

    return data


def load_multifile_timeseries(bin_paths, chans, datetime=False, xarray=False):
    """Load and concatenate multiple SpikeGLX files.
    Will load all data into memory unless using xarray.
    If not using xarray, all data are assumed to be contiguous without gaps.
    """

    all_data = [
        load_timeseries(
            path,
            chans,
            start_time=None,
            end_time=None,
            datetime=datetime,
            xarray=xarray,
        )
        for path in bin_paths
    ]

    if xarray:
        return xr.concat(all_data, dim="time")
    else:
        print(
            "You are loading multifile SGLX data without xarray.\n",
            "Are you sure you want to do this? Please see documentation.",
        )
        all_time, all_sig, all_fs = zip(*all_data)
        file_durations = [time.max() for time in all_time]
        file_ends = np.cumsum(file_durations)
        offset_time = np.concatenate(
            [all_time[i] + file_ends[i - 1] for i in range(1, len(all_time))]
        )
        time = np.concatenate([all_time[0], offset_time])
        sig = np.concatenate(all_sig)
        fs = all_fs[0]
        assert np.all(
            np.asarray(all_fs) == fs
        ), "All recordings must have the same sampling rate"
        return (time, sig, fs)
