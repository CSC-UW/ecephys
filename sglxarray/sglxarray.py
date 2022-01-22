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
    # Calculate file's start and end samples
    nFileChan = int(meta["nSavedChans"])
    nFileSamp = int(int(meta["fileSizeBytes"]) / (2 * nFileChan))
    (firstFileSamp, lastFileSamp) = (0, nFileSamp - 1)

    # Get the start and end samples
    firstSample = int(max(firstFileSamp, firstSample))
    lastSample = int(min(lastFileSamp, lastSample))

    return firstSample, lastSample


def _get_timestamps(meta, firstSample=0, lastSample=np.Inf):
    """Get all timestamps contained in the data."""

    firstSample, lastSample = _get_first_and_last_samples(meta, firstSample, lastSample)

    # Get timestamps of each sample
    fs = SampRate(meta)
    time = np.arange(firstSample, lastSample + 1)
    time = time / fs  # timestamps in seconds from start of file
    timedelta = pd.to_timedelta(time, "s")  # as timedelta objects

    datetime = pd.to_datetime(meta["fileCreateTime"]) + timedelta

    return time, timedelta, datetime


def _to_seconds(t, meta):
    """Convert any time into seconds from the start of the file.
    See `start_time` and `end_time` arguments to `load_trigger` for
    expected behavior and accepted types."""
    if isinstance(t, str):
        _, _, dt = _get_timestamps(meta)
        i = _find_nearest(dt._get_time_micros(), _time_to_micros(to_time(t)))
        t = dt[i]

    if is_datetime64_any_dtype(t) or isinstance(t, pd.Timestamp):
        t0 = pd.to_datetime(meta["fileCreateTime"])
        return (t - t0).total_seconds()
    if is_timedelta64_dtype(t) or isinstance(t, pd.Timedelta):
        return t.total_seconds()

    if is_datetime_or_timedelta_dtype(t):
        raise ValueError("Unexpected datetime or timedelta object type.")

    if isinstance(t, numbers.Real):
        return t

    raise ValueError(f"Could not convert {t} to time.")


def load_trigger(bin_path, channels=None, start_time=0, end_time=np.Inf):
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
        - If float: Time relative to the file start, in seconds.
        - If timedelta: Time relative to the file start.
        - If datetime: Any absolute datetime is fine.
        - If string: Time of day, e.g. '13:00:00' for 1PM.
            Similar to DataFrame.at_time, but rounds to the nearest sample in the data.
        - If the value provided works out to be less than 0 or before the actual file start time,
            the real file start time will be used.
        - Default: 0.0, a.k.a. the start of the file.
    end_time: float, timedelta, datetime, or string (optional)
        End time of the data to load.
        - The behavior is the same as for `start_time`, but if  the value provided works out
            to be greater than the actual file end time, the actual file end time will be used.
        - Default: np.Inf, a.k.a. the end of the file.

    Returns
    -------
    data : xr.DataArray (n_samples, n_chans)
        Dimensions:
            time: float
                Time in seconds from the start of the file.
            channel: int
                Channel IDs of the loaded data.
        Coordinates:
            (time, timedelta): timedelta64[ns]
                Timedelta in seconds from the start of the file, nanosecond resolution.
            (time, datetime): datetime64[ns]
                Absolute timestamp of each sample, estimated using the `fileCreateTime`
                    field from the metadata. Nanosecond resolution.
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
    firstSamp = _to_seconds(start_time, meta) * fs
    lastSamp = _to_seconds(end_time, meta) * fs

    # Get the start and end samples
    firstSamp, lastSamp = _get_first_and_last_samples(meta, firstSamp, lastSamp)

    # Get timestamps of each sample
    time, timedelta, datetime = _get_timestamps(meta, firstSamp, lastSamp)

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
            "timedelta": ("time", timedelta),
            "datetime": ("time", datetime),
            "x": ("channel", im.chans2coords(channels)[:, 0]),
            "y": ("channel", im.chans2coords(channels)[:, 1]),
        },
        attrs={"units": sig_units, "fs": fs, "im": im},
    )

    return data


def load_contiguous_triggers(bin_paths, chans=None):
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
    triggers = [load_trigger(path, chans) for path in bin_paths]
    data = xr.concat(triggers, dim="time")

    time = np.arange(data.time.size) / data.fs
    timedelta = pd.to_timedelta(time)
    datetime = data.datetime.values.min() + timedelta

    return data.assign_coords(
        {"time": time, "timedelta": ("time", timedelta), "datetime": ("time", datetime)}
    )
