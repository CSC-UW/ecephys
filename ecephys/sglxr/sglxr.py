import datetime
import pathlib
from typing import Optional

import dask
import dask.array as da
import numpy as np
import xarray as xr
import pandas as pd
import numbers
import pandas.core.tools.times
from pandas.core.dtypes.common import (
    is_datetime64_any_dtype,
    is_timedelta64_dtype,
    is_datetime_or_timedelta_dtype,
)

from ecephys.sglxr import ImecMap
from ecephys.sglxr.external import readSGLX


def validate_probe_type(meta: dict):
    if ("imDatPrb_type" not in meta) or (int(meta["imDatPrb_type"]) != 0):
        raise NotImplementedError(
            "This module has only been tested with Neuropixel 1.0 probes."
        )


def _find_nearest(array: np.ndarray, value, tie_select="first") -> int:
    """Index of element in array nearest to value.

    Return either first or last value if ties"""
    array = np.asarray(array)
    a = np.abs(array - value)
    if tie_select == "first":
        return a.argmin()
    elif tie_select == "last":
        # reverse array to find last occurence
        b = a[::-1]
        return len(b) - np.argmin(b) - 1
    else:
        raise ValueError()


def _time_to_micros(time_obj: datetime.time) -> float:
    """Convert datetime.time to total microseconds.
    Taken from pandas/core/indexes/datetimes.py"""
    seconds = time_obj.hour * 60 * 60 + 60 * time_obj.minute + time_obj.second
    return 1_000_000 * seconds + time_obj.microsecond


def _get_first_and_last_samples(
    meta: dict, firstSample: int = 0, lastSample: int = np.Inf
) -> tuple[int, int]:
    """Take requested start/end sample numbers, and
    return the closest actual start/end sample numbers."""
    # Calculate file's start and end samples
    nFileChan = int(meta["nSavedChans"])
    nFileSamp = int(int(meta["fileSizeBytes"]) / (2 * nFileChan))
    (firstFileSamp, lastFileSamp) = (0, nFileSamp - 1)

    # Get the start and end samples
    firstSample = int(max(firstFileSamp, firstSample))
    lastSample = int(min(lastFileSamp, lastSample))

    return firstSample, lastSample


def _get_timestamps(
    meta: dict,
    firstSample: int = 0,
    lastSample: int = np.Inf,
    t0: float = 0.0,
    dt0="fileCreateTime",
) -> tuple[np.ndarray, pd.DatetimeIndex, float]:
    """Get all timestamps contained in the data."""

    firstSample, lastSample = _get_first_and_last_samples(meta, firstSample, lastSample)

    # Get timestamps of each sample
    fs = readSGLX.SampRate(meta)
    time = np.arange(firstSample, lastSample + 1)
    time = time / fs  # timestamps in seconds from start of file
    timedelta = pd.to_timedelta(time, "s")  # as timedelta objects

    if dt0 == "fileCreateTime":
        dt0 = pd.to_datetime(meta["fileCreateTime"])

    return t0 + time, dt0 + timedelta, fs


def get_timestamps(
    bin_path: pathlib.Path, **kwargs
) -> tuple[np.ndarray, pd.DatetimeIndex, float]:
    return _get_timestamps(readSGLX.readMeta(pathlib.Path(bin_path)), **kwargs)


def _to_seconds_from_file_start(x, meta: dict, **kwargs) -> float:
    """Convert any time into seconds from the start of the file.
    See `start_time` and `end_time` arguments to `load_trigger` for
    expected behavior and accepted types."""
    t, dt, _ = _get_timestamps(meta, **kwargs)

    if isinstance(x, str):
        x = dt[
            _find_nearest(
                dt._get_time_micros(),
                _time_to_micros(pandas.core.tools.times.to_time(x)),
            )
        ]

    if is_datetime64_any_dtype(x) or isinstance(x, pd.Timestamp):
        return (x - dt.min()).total_seconds()
    if is_timedelta64_dtype(x) or isinstance(x, pd.Timedelta):
        return x.total_seconds()

    if is_datetime_or_timedelta_dtype(x):
        raise ValueError("Unexpected datetime or timedelta object type.")

    if isinstance(x, numbers.Real):
        return x - t.min()

    raise ValueError(f"Could not convert {x} to time.")


def _memmap_and_load_chunk(
    binpath: pathlib.Path, nChan: int, nFileSamp: int, sl: slice
) -> np.ndarray:
    data = np.memmap(
        binpath, mode="r", shape=(nChan, nFileSamp), dtype="int16", offset=0, order="F"
    )
    return data[:, sl]


def memmap_dask_array(
    binpath: pathlib.Path, meta: dict, blocksize: int = 250000
) -> da.Array:
    """Returns a dask array backed by a memory map of the binary file.
    Shape is channels x time, chunked along the time dimension.
    If blocksize is -1, the entire file is loaded into a single chunk.
    See https://docs.dask.org/en/stable/array-creation.html"""
    nChan = int(meta["nSavedChans"])
    nFileSamp = int(int(meta["fileSizeBytes"]) / (2 * nChan))
    if blocksize == -1:
        blocksize = nFileSamp
    print("nChan: %d, nFileSamp: %d" % (nChan, nFileSamp))
    load = dask.delayed(_memmap_and_load_chunk)
    chunks = []
    for index in range(0, nFileSamp, blocksize):
        # Truncate the last chunk if necessary
        chunk_size = min(blocksize, nFileSamp - index)
        chunk = dask.array.from_delayed(
            load(
                binpath,
                nChan=nChan,
                nFileSamp=nFileSamp,
                sl=slice(index, index + chunk_size),
            ),
            shape=(nChan, chunk_size),
            dtype="int16",
        )
        chunks.append(chunk)
    return da.concatenate(chunks, axis=1)


def _convert_to_uv(data: da.Array, channels: np.ndarray[int], meta: dict) -> da.Array:
    """Takes (channel, time) raw int16 data and converts to (time, channel) float data in uV."""
    # Look up gain with acquired channel ID
    chans = readSGLX.OriginalChans(meta)
    APgain, LFgain = readSGLX.ChanGainsIM(meta)
    nAP = len(APgain)
    nNu = nAP * 2  # num neural channels

    # Common conversion factor
    fI2V = readSGLX.Int2Volts(meta)

    # Create array of conversion factors for each channel
    convArray = np.zeros(data.shape[0], dtype="float")
    for i in range(0, len(channels)):
        j = channels[i]  # chan id
        k = chans[j]  # acquisition index
        if k < nAP:  # If this is an AP channel, apply AP gain
            conv = fI2V / APgain[k]
        elif k < nNu:  # If this is an LF channel, apply LF gain
            conv = fI2V / LFgain[k - nAP]
        else:  # Otherwise, no gain
            conv = 1
        convArray[i] = conv
    # Apply convertion to volts, then convert to microvolts
    return 1e6 * (data.T * convArray)


# TODO: Also return a `sample` coord on the time dimension?
def open_trigger(
    bin_path: pathlib.Path,
    channels: list[int] = None,
    start_time: float = 0,
    end_time: float = np.Inf,
    t0: float = 0.0,
    dt0="fileCreateTime",
    blocksize: int = 250000,
) -> xr.DataArray:
    """Open SpikeGLX timeseries data as an xarray DataArray, backed by a lazy dask array.

    Parameters
    ----------
    bin_path: Path object
        The path to the binary data (i.e. *.bin).
    chans: array of int, None
        The list of channel IDs to load. These are probably what you would intuitively
        provide, but they are NOT the same thing as the electrode/site numbers, the
        acquistion orders, or the channel map user orders!
        - If `None`, load all channels found in the metadata.
        - Default: None, a.k.a. load all channels.
    start_time: float, timedelta, datetime, or string (optional)
        Start time of the data to load.
        - If float: Absolute time (i.e. including t0), in seconds.
        - If timedelta: Time relative to the file start.
        - If datetime: Any absolute datetime is fine.
        - If string: Time of day, e.g. '13:00:00' for 1PM.
            Similar to DataFrame.at_time, but rounds to the nearest sample in the data.
        - If the value provided works out to be less than 0 or before the actual file start time,
            the real file start time will be used.
        - Default: 0.0, aka. the start of the file when t0 is 0.0
    end_time: float, timedelta, datetime, or string (optional)
        End time of the data to load.
        - The behavior is the same as for `start_time`, but if  the value provided works out
            to be greater than the actual file end time, the actual file end time will be used.
        - Default: np.Inf, a.k.a. the end of the file.
    t0: float (optional)
        Force the first timestamp in the file (not necessarily the loaded data) to this value.
        Default: 0.0
    dt0: datetime (optional) or 'fileCreateTime'
        Force the first datetime stamp in the file (not necessarily the loaded data) to this value. If 'fileCreateTime', use metadata.
        Default: 'fileCreateTime'
    blocksize: int
        The desired chunk size, in samples, of the data.
        If -1, load the entire file as a single chunk.

    Returns
    -------
    data : xr.DataArray (n_samples, n_chans)
        Dimensions:
            time: float
                Time in seconds from the start of the file.
            channel: int
                Channel IDs of the loaded data.
        Coordinates:
            (time, datetime): datetime64[ns]
                Absolute timestamp of each sample, nanosecond resolution.
            (channel, x): float
                X coordinate in probe space, in microns, of each channel.
            (channel, y): float
                Y coorindate in probe space, in microns, of each channel.
        Attrs:
            units: string
                The units of the loaded data, i.e. uV
            fs: float
                The sampling rate of the data.
            im: ImecMap
                The full IMRO + channel map for the data.
    """
    # Read and validate the metadata
    bin_path = pathlib.Path(bin_path)
    meta = readSGLX.readMeta(bin_path)
    validate_probe_type(meta)

    # Get the requested start and end samples
    fs = readSGLX.SampRate(meta)
    firstSamp = _to_seconds_from_file_start(start_time, meta, t0=t0, dt0=dt0) * fs
    lastSamp = _to_seconds_from_file_start(end_time, meta, t0=t0, dt0=dt0) * fs

    # Get the start and end samples
    firstSamp, lastSamp = _get_first_and_last_samples(meta, firstSamp, lastSamp)

    # Get timestamps of each sample
    time, datetime, _ = _get_timestamps(meta, firstSamp, lastSamp, t0=t0, dt0=dt0)

    # Make memory map to selected data.
    im = ImecMap.from_meta(meta)
    channels = im.chans if channels is None else channels
    rawData = memmap_dask_array(bin_path, meta, blocksize)
    selectData = rawData[channels, firstSamp : lastSamp + 1]

    # apply gain correction and convert to uV
    assert (
        meta["typeThis"] == "imec"
    ), "This function only supports loading of analog IMEC data."
    sig = _convert_to_uv(selectData, channels, meta)
    sig_units = "uV"

    # Wrap data with xarray
    data = xr.DataArray(
        sig,
        dims=("time", "channel"),
        coords={
            "time": time,
            "channel": channels,
            "datetime": ("time", datetime),
            "x": ("channel", np.atleast_2d(im.chans2coords(channels))[:, 0]),
            "y": ("channel", np.atleast_2d(im.chans2coords(channels))[:, 1]),
        },
        attrs={"units": sig_units, "fs": fs, "im": im},
        name="traces",
    )

    return data


def load_trigger(*args, **kwargs) -> xr.DataArray:
    return open_trigger(*args, **kwargs).compute()
