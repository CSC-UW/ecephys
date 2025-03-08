from pathlib import Path

import numpy as np
import xarray as xr

# TODO: sglxr should not be a dependency of sglx
from ecephys.sglxr import sglxr

# TODO: This should be satisfied by sglx.external
from ecephys.sglxr.external import readSGLX


def _is_channel_XA(meta, channel):
    MN, MA, XA, DW = readSGLX.ChannelCountsNI(meta)
    return (channel >= (MN + MA)) and (channel < XA)


def load_nidq_analog(bin_path, channels, start_time=0, end_time=np.Inf):
    # Read and validate the metadata
    bin_path = Path(bin_path)
    meta = readSGLX.readMeta(bin_path)

    # Get the requested start and end samples
    fs = readSGLX.SampRate(meta)
    firstSamp = sglxr._to_seconds_from_file_start(start_time, meta) * fs
    lastSamp = sglxr._to_seconds_from_file_start(end_time, meta) * fs

    # Get the start and end samples
    firstSamp, lastSamp = sglxr._get_first_and_last_samples(meta, firstSamp, lastSamp)

    # Get timestamps of each sample
    time, datetime, _ = sglxr._get_timestamps(meta, firstSamp, lastSamp)

    # Make memory map to selected data.
    rawData = readSGLX.makeMemMapRaw(bin_path, meta)
    selectData = rawData[channels, firstSamp : lastSamp + 1]

    # Apply gain correction and convert to V
    assert meta["typeThis"] == "nidq", (
        "This function only supports loading of analog NIDQ data."
    )
    assert all(_is_channel_XA(meta, ch) for ch in channels), (
        "This function only supports loading of analog NIDQ data."
    )
    sig = 1e3 * readSGLX.GainCorrectNI(selectData, channels, meta)
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
