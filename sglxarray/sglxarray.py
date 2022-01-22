import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path

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


def load_timestamps(bin_path, start_time=None, end_time=None):
    """Load SpikeGLX timestamps

    Parameters
    ----------
    bin_path: Path object
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
    bin_path = Path(bin_path)
    meta = readMeta(bin_path)
    validate_probe_type(meta)
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


def load_trigger(bin_path, chans=None, start_time=0, end_time=np.Inf):
    """Load SpikeGLX timeseries data.

    Parameters
    ----------
    bin_path: Path object
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
        Attrs: units, fs
    """
    bin_path = Path(bin_path)
    meta = readMeta(bin_path)
    validate_probe_type(meta)

    im = ImecMap.from_meta(meta)
    chans = im.chans if chans is None else chans

    rawData = makeMemMapRaw(bin_path, meta)

    # Calculate file's start and end samples
    nFileChan = int(meta["nSavedChans"])
    nFileSamp = int(int(meta["fileSizeBytes"]) / (2 * nFileChan))
    (firstFileSamp, lastFileSamp) = (0, nFileSamp - 1)

    # Get the requested start and end samples
    fs = SampRate(meta)
    firstRequestedSamp = fs * start_time
    lastRequestedSamp = fs * end_time

    # Get the start and end samples
    firstSamp = int(max(firstFileSamp, firstRequestedSamp))
    lastSamp = int(min(lastFileSamp, lastRequestedSamp))

    # Get timestamps of each sample
    time = np.arange(firstSamp, lastSamp + 1)
    time = time / fs  # timestamps in seconds from start of file
    timedelta = pd.to_timedelta(time, "s")
    datetime = pd.to_datetime(meta["fileCreateTime"]) + timedelta

    selectData = rawData[chans, firstSamp : lastSamp + 1]

    # apply gain correction and convert to uV
    assert (
        meta["typeThis"] == "imec"
    ), "This function only supports loading of analog IMEC data."
    sig = 1e6 * GainCorrectIM(selectData, chans, meta)
    sig_units = "uV"

    # Wrap data with xarray
    data = xr.DataArray(
        sig.T,
        dims=("time", "channel"),
        coords={
            "time": time,
            "channel": chans,
            "timedelta": ("time", timedelta),
            "datetime": ("time", datetime),
            "x": ("channel", im.chans2coords(chans)[:, 0]),
            "y": ("channel", im.chans2coords(chans)[:, 1]),
        },
        attrs={"units": sig_units, "fs": fs, "ImecMap": im},
    )

    return data


def load_contiguous_triggers(bin_paths, chans):
    """Load and concatenate a list of contiguous SGLX files.

    Parameters
    ----------
    bin_paths: iterable Path objects
        The data to concatenate, in order.
    chans: 1d array
        The list of channels to load.

    Returns
    -------
    data: xr.DataArray
        The concatenated data.
        Metadata is copied from the first file.
    """
    triggers = [load_trigger(path, chans) for path in bin_paths]
    data = xr.concat(triggers, dim="time")

    time = np.arange(data.time.size) / data.fs
    timedelta = pd.to_timedelta(time)
    datetime = data.datetime.values.min() + timedelta

    return data.assign_coords(
        {"time": time, "timedelta": ("time", timedelta), "datetime": ("time", datetime)}
    )
