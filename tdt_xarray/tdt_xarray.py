import numpy as np
import xarray as xr
import pandas as pd
import tdt
from collections.abc import Iterable


def load_store_names(block_path):
    """Get the names of each store in the block."""
    hdr = tdt.read_block(block_path, headers=True)
    return [store.name for key, store in hdr.stores.items()]


def load_stream_names(block_path):
    """Get the names of each stream store in the block."""
    hdr = tdt.read_block(block_path, headers=True)
    return [
        store.name for key, store in hdr.stores.items() if store.type_str == "streams"
    ]


def load_channel_lists(block_path):
    """Get the list of channels associated with each stream store."""
    hdr = tdt.read_block(block_path, headers=True)
    return {
        store.name: list(np.unique(store.chan))
        for key, store in hdr.stores.items()
        if hasattr(store, "chan")
    }


def _load_stream_store(block_path, store_name, chans=None, start_time=0, end_time=0):
    """Load a single stream store from disk.

    Parameters:
    -----------
    block_path:
        Path to the TDT block directory which contains files of type .Tbk, .tev, .tsq, etc.
    store_name: string
        The name of the stream store to load.
    chans: Iterable, optional, default: None
        The list of channels to load, as they appear in Synapse/OpenEx. These will be 1-indexed,
        rather than 0-indexed. Passing `None` (default) loads all channels.
    start_time: float, optional, default: 0
        Start time of the data to load, relative to the file start, in seconds.
    end_time: float, optional, default: 0
        End time of the data to load, relative to the file start, in seconds.
        Passing `0` (default) will load until the end of the file.

    Returns:
    --------
    info: tdt.StructType
        The `info` field of a tdt `blk` struct, used to get the file start time.
    store: tdt.StructType
        The store field of a tdt `blk.streams` struct, which contains the data.
    """
    assert store_name not in [
        "epocs",
        "snips",
        "streams",
        "scalars",
        "info",
        "time_ranges",
    ]

    read_block_kwargs = dict(store=["info", store_name], t1=start_time, t2=end_time)
    if chans:
        assert isinstance(chans, Iterable)
        assert 0 not in chans, "Passing 0 to tdt.read_block will load all channels."
        read_block_kwargs.update(channel=chans)

    blk = tdt.read_block(block_path, **read_block_kwargs)
    return blk.info, blk.streams[store_name]


def stream_store_to_xarray(info, store):
    """Convert a single stream store to xarray format.

    Paramters:
    ----------
    info: tdt.StructType
        The `info` field of a tdt `blk` struct, as returned by `_load_stream_store`.
    store: tdt.StructType
        The store field of a tdt `blk.streams` struct, as returned by `_load_stream_store`.

    Returns:
    --------
    data: xr.DataArray (n_samples, n_channels)
        Values: The data, in microvolts.
        Attrs: units, fs
        Name: The store name
    """
    n_channels, n_samples = np.atleast_2d(store.data).shape

    time = np.arange(0, n_samples) / store.fs + store.start_time
    timedelta = pd.to_timedelta(time, "s")
    datetime = pd.to_datetime(info.start_date) + timedelta

    # SEVs use 'channels', while TEVs use 'channel'.
    if ("channel" in store.keys()) and not ("channels" in store.keys()):
        channels = store.channel
    elif ("channels" in store.keys()) and not ("channel" in store.keys()):
        channels = store.channels
    else:
        ValueError("Stream store should contain 'channel' or 'channels' fields.")

    volts_to_microvolts = 1e6
    data = xr.DataArray(
        np.atleast_2d(store.data).T * volts_to_microvolts,
        dims=("time", "channel"),
        coords={
            "time": time,
            "channel": channels,
            "timedelta": ("time", timedelta),
            "datetime": ("time", datetime),
        },
        name=store.name,
    )
    data.attrs["units"] = "uV"
    data.attrs["fs"] = store.fs

    return data


def load_stream_store(*args, **kwargs):
    """Load a single stream store from disk and convert to xarray format.

    Parameters:
    -----------
    block_path:
        Path to the TDT block directory which contains files of type .Tbk, .tev, .tsq, etc.
    store_name: string
        The name of the stream store to load.
    chans: Iterable, optional, default: None
        The list of channels to load, as they appear in Synapse/OpenEx. These will be 1-indexed,
        rather than 0-indexed. Passing `None` (default) loads all channels.
    start_time: float, optional, default: 0
        Start time of the data to load, relative to the file start, in seconds.
    end_time: float, optional, default: 0
        End time of the data to load, relative to the file start, in seconds.
        Passing `0` (default) will load until the end of the file.

    Returns:
    --------
    data: xr.DataArray (n_samples, n_channels)
        Values: The data, in microvolts.
        Attrs: units, fs
        Name: The store name
    """
    info, store = _load_stream_store(*args, **kwargs)
    return stream_store_to_xarray(info, store)


def load_all_streams(block_path, start_time=0, end_time=0):
    """Load all stream stores in a block."""
    store_names = load_stream_names(block_path)
    store_data = dict()
    for name in store_names:
        store_data[name] = load_stream_store(
            block_path, name, start_time=start_time, end_time=end_time
        )

    return store_data
