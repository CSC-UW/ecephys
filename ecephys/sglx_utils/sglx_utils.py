import os.path
from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd

from .external import SGLXMetaToCoords
from .external.readSGLX import (
    makeMemMapRaw,
    readMeta,
    SampRate,
    GainCorrectIM,
)


def get_xy_coords(binpath):
    """Return AP channel indices and their x and y coordinates."""
    metapath = Path(binpath).with_suffix(".meta")
    chans, xcoord, ycoord = SGLXMetaToCoords.MetaToCoords(metapath, 4)
    return chans, xcoord, ycoord


def load_timestamps(bin_path, start_time=None, end_time=None):
    """Load SpikeGLX timestamps

    Parameters
    ----------
    bin_path: joblib Path object
        The path to the binary data (i.e. *.bin)
    start_time: float, optional, default: None
        Start time of the data to load, relative to the file start, in seconds.
        If `None`, load from the start of the file.
    end_time: float, optional, default: None
        End time of the data to load, relative to the file start, in seconds.
        If `None`, load until the end of the file.

    Returns
    -------
    time : np.array (n_samples, )
        Time of each sample, in seconds.
    """
    meta = readMeta(bin_path)
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

    return time


def load_timeseries(bin_path, chans, start_time=0, end_time=np.Inf):
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
    data : xr.DataArray (n_samples, n_chans)
        Attrs: units, fs, fileCreateTime, firstSample
    """

    meta = readMeta(bin_path)
    rawData = makeMemMapRaw(bin_path, meta)
    fs = SampRate(meta)

    # Calculate file's start and end samples
    nFileChan = int(meta["nSavedChans"])
    nFileSamp = int(int(meta["fileSizeBytes"]) / (2 * nFileChan))
    (firstFileSamp, lastFileSamp) = (0, nFileSamp - 1)

    # Get the requested start and end samples
    firstRequestedSamp = int(fs * start_time)
    lastRequestedSamp = int(fs * end_time)

    # Get the start and end samples
    firstSamp = max(firstFileSamp, firstRequestedSamp)
    lastSamp = min(lastFileSamp, lastRequestedSamp)

    # Get timestamps of each sample
    time = np.arange(firstSamp, lastSamp + 1)
    time = time / fs  # timestamps in seconds from start of file

    selectData = rawData[chans, firstSamp : lastSamp + 1]

    # apply gain correction and convert to uV
    assert (
        meta["typeThis"] == "imec"
    ), "This function only supports loading of analog IMEC data."
    sig = 1e6 * GainCorrectIM(selectData, chans, meta)
    sig_units = "uV"

    # Wrap data with xarray
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