from multiprocessing.sharedctypes import Value
import numpy as np
import xarray as xr
import pandas as pd
import numbers
from pathlib import Path
from pandas.core.tools.times import to_time
from pandas.core.dtypes.common import (
    is_datetime64_any_dtype,
    is_timedelta64_dtype,
    is_datetime_or_timedelta_dtype,
)

from .imec_map import ImecMap
from .external.readSGLX import (
    makeMemMapRaw,
    readMeta,
    SampRate,
    GainCorrectIM,
)


def validate_probe_type(meta):
    if ("imDatPrb_type" not in meta) or (int(meta["imDatPrb_type"]) != 0):
        raise NotImplementedError(
            "This module has only been tested with Neuropixel 1.0 probes."
        )


def _find_nearest(array, value, tie_select="first"):
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


def _time_to_micros(time_obj):
    """Convert datetime.time to total microseconds.
    Taken from pandas/core/indexes/datetimes.py"""
    seconds = time_obj.hour * 60 * 60 + 60 * time_obj.minute + time_obj.second
    return 1_000_000 * seconds + time_obj.microsecond


def _get_first_and_last_samples(meta, firstSample=0, lastSample=np.Inf):
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
    meta, firstSample=0, lastSample=np.Inf, t0=0.0, dt0="fileCreateTime"
):
    """Get all timestamps contained in the data."""

    firstSample, lastSample = _get_first_and_last_samples(meta, firstSample, lastSample)

    # Get timestamps of each sample
    fs = SampRate(meta)
    time = np.arange(firstSample, lastSample + 1)
    time = time / fs  # timestamps in seconds from start of file
    timedelta = pd.to_timedelta(time, "s")  # as timedelta objects

    if dt0 == "fileCreateTime":
        dt0 = pd.to_datetime(meta["fileCreateTime"])

    return t0 + time, dt0 + timedelta, fs


def get_timestamps(bin_path, **kwargs):
    return _get_timestamps(readMeta(Path(bin_path)), **kwargs)


def _to_seconds_from_file_start(x, meta, **kwargs):
    """Convert any time into seconds from the start of the file.
    See `start_time` and `end_time` arguments to `load_trigger` for
    expected behavior and accepted types."""
    t, dt, _ = _get_timestamps(meta, **kwargs)

    if isinstance(x, str):
        x = dt[_find_nearest(dt._get_time_micros(), _time_to_micros(to_time(x)))]

    if is_datetime64_any_dtype(x) or isinstance(x, pd.Timestamp):
        return (x - dt.min()).total_seconds()
    if is_timedelta64_dtype(x) or isinstance(x, pd.Timedelta):
        return x.total_seconds()

    if is_datetime_or_timedelta_dtype(x):
        raise ValueError("Unexpected datetime or timedelta object type.")

    if isinstance(x, numbers.Real):
        return x - t.min()

    raise ValueError(f"Could not convert {x} to time.")


# TODO: Also return a `sample` coord on the time dimension
def load_trigger(
    bin_path, channels=None, start_time=0, end_time=np.Inf, t0=0.0, dt0="fileCreateTime"
):
    """Load SpikeGLX timeseries data.

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
    bin_path = Path(bin_path)
    meta = readMeta(bin_path)
    validate_probe_type(meta)

    # Get the requested start and end samples
    fs = SampRate(meta)
    firstSamp = _to_seconds_from_file_start(start_time, meta, t0=t0, dt0=dt0) * fs
    lastSamp = _to_seconds_from_file_start(end_time, meta, t0=t0, dt0=dt0) * fs

    # Get the start and end samples
    firstSamp, lastSamp = _get_first_and_last_samples(meta, firstSamp, lastSamp)

    # Get timestamps of each sample
    time, datetime, _ = _get_timestamps(meta, firstSamp, lastSamp, t0=t0, dt0=dt0)

    # Make memory map to selected data.
    im = ImecMap.from_meta(meta)
    channels = im.chans if channels is None else channels
    rawData = makeMemMapRaw(bin_path, meta)
    selectData = rawData[channels, firstSamp : lastSamp + 1]

    # apply gain correction and convert to uV
    assert (
        meta["typeThis"] == "imec"
    ), "This function only supports loading of analog IMEC data."
    sig = 1e6 * GainCorrectIM(selectData, channels, meta)
    sig_units = "uV"

    # Wrap data with xarray
    data = xr.DataArray(
        sig.T,
        dims=("time", "channel"),
        coords={
            "time": time,
            "channel": channels,
            "datetime": ("time", datetime),
            "x": ("channel", np.atleast_2d(im.chans2coords(channels))[:, 0]),
            "y": ("channel", np.atleast_2d(im.chans2coords(channels))[:, 1]),
        },
        attrs={"units": sig_units, "fs": fs, "im": im},
    )

    return data


def load_contiguous_triggers(bin_paths, chans=None, t0=0.0, dt0="fileCreateTime"):
    """Load and concatenate a list of temporally contiguous SGLX files.

    Parameters
    ----------
    bin_paths: iterable Path objects
        The data to concatenate, in order.
    chans:
        See `load_trigger`.

    Returns
    -------
    data: xr.DataArray
        The concatenated data. See `load_trigger` for details.
        Metadata is copied from the first file.
        Datetimes are rebased so that the `fileCreateTime` field of the
            very first file's metadata is used as t0
    """
    triggers = [load_trigger(bin_paths[0], chans, t0=t0, dt0=dt0)] + [
        load_trigger(p, chans) for p in bin_paths[1:]
    ]
    data = xr.concat(triggers, dim="time")

    time = np.arange(data.time.size) / data.fs
    timedelta = pd.to_timedelta(time, "s")

    return data.assign_coords(
        {
            "time": data.time.values.min() + time,
            "datetime": ("time", data.datetime.values.min() + timedelta),
        }
    )
