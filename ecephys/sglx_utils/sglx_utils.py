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

    data = xr.DataArray(
        sig.T, dims=("time", "channel"), coords={"time": time, "channel": chans}
    )
    data.attrs["units"] = sig_units
    data.attrs["fs"] = fs
    data.attrs["fileCreateTime"] = meta["fileCreateTime"]
    data.attrs["firstSample"] = meta["firstSample"]

    return data


def load_multifile_timeseries(bin_paths, chans, contiguous=False):
    """Load and concatenate multiple SpikeGLX files.

    Parameters
    ----------
    bin_paths: iterable Path objects
        The data to concatenate, in order.
    chans: 1d array
        The list of channels to load.
    contiguous: bool
        If `true`, data are assumed to be contiguous (i.e. no gaps)

    Returns
    -------
    data: xr.DataArray
        The concatenated data.
        If contiguous, "time" dimension is a numpy array of timestamps in seconds.
        If not contiguous, "time" dimension is an array of datetime objects.
        Metadata is copied from the first file.
    """

    all_data = [
        load_timeseries(
            path,
            chans,
            start_time=None,
            end_time=None,
        )
        for path in bin_paths
    ]

    all_fs = [data.fs for data in all_data]
    fs = all_fs[0]
    assert np.all(
        np.asarray(all_fs) == fs
    ), "All recordings must have the same sampling rate"

    if contiguous:
        samples_per_file = np.asarray([len(data) for data in all_data])
        total_samples = np.sum(samples_per_file)
        time = np.arange(0, total_samples) / fs
        first_samples = np.cumsum(samples_per_file) - samples_per_file
        last_samples = np.cumsum(samples_per_file)

        for i in range(0, len(all_data)):
            all_data[i]["time"] = time[first_samples[i] : last_samples[i]]
    else:
        for data in all_data:
            data["time"] = pd.to_datetime(data.fileCreateTime) + pd.to_timedelta(
                data.time.values, "s"
            )

    return xr.concat(all_data, dim="time")
