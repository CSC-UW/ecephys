import numpy as np
import xarray as xr
from ecephys.sglxr.sglxr import (
    _to_seconds_from_file_start,
    _get_first_and_last_samples,
    _get_timestamps,
)
from pathlib import Path
from ecephys.sglxr.external.readSGLX import (
    ChannelCountsNI,
    readMeta,
    SampRate,
    makeMemMapRaw,
    GainCorrectNI,
)


def is_channel_XA(meta, channel):
    MN, MA, XA, DW = ChannelCountsNI(meta)
    return (channel >= (MN + MA)) and (channel < XA)


def load_nidq_analog(bin_path, channels, start_time=0, end_time=np.Inf):
    # Read and validate the metadata
    bin_path = Path(bin_path)
    meta = readMeta(bin_path)

    # Get the requested start and end samples
    fs = SampRate(meta)
    firstSamp = _to_seconds_from_file_start(start_time, meta) * fs
    lastSamp = _to_seconds_from_file_start(end_time, meta) * fs

    # Get the start and end samples
    firstSamp, lastSamp = _get_first_and_last_samples(meta, firstSamp, lastSamp)

    # Get timestamps of each sample
    time, datetime, _ = _get_timestamps(meta, firstSamp, lastSamp)

    # Make memory map to selected data.
    rawData = makeMemMapRaw(bin_path, meta)
    selectData = rawData[channels, firstSamp : lastSamp + 1]

    # Apply gain correction and convert to V
    assert (
        meta["typeThis"] == "nidq"
    ), "This function only supports loading of analog NIDQ data."
    assert all(
        is_channel_XA(meta, ch) for ch in channels
    ), "This function only supports loading of analog NIDQ data."
    sig = 1e3 * GainCorrectNI(selectData, channels, meta)
    sig_units = "mV"

    # Wrap data with xarray
    return xr.DataArray(
        sig.T,
        dims=("time", "channel"),
        coords={
            "time": time,
            "channel": channels,
            "datetime": ("time", datetime),
        },
        attrs={"units": sig_units, "fs": fs},
    )
