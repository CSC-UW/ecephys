"""Load data in multiple formats as memmaps or arrays."""
import os.path
from pathlib import Path

import numpy as np

import tdt

from . import readSGLX as SGLX
from . import resample, utils

DATA_FORMATS = ['SGLX', 'OpenEphys', 'TDT']


def loader_switch(binPath, *args, datatype='SGLX', **kwargs):
    """Pipe to correct function for array loading.

    Args:
        binPath (str or pathlib.Path): Path to binary data
        datatype (str): 'SGLX' or 'OpenEphys' (default 'SGLX')
        *args: Passed to loading function for the considered data format

    Kwargs:
        *kwargs: Passed to loading function for the considered data format
    """
    if datatype not in DATA_FORMATS:
        raise ValueError(
            f'Data format: `{datatype}` not supported.\n'
            f'Supported values for `datatype` parameter: {DATA_FORMATS}'
        )
    if datatype == 'OpenEphys':
        raise NotImplementedError

    if not os.path.exists(binPath):
        raise FileNotFoundError(f"No file at binPath: `{binPath}`")

    data, sf, channels = LOADING_FUNCTIONS[datatype.lower()](
        binPath,
        **kwargs
    )

    print_loading_output(binPath, data, sf, channels)
    return data, sf, channels


def print_loading_output(binPath, data, sf, channels):
    info = ("Data successfully loaded (%s):"
            "\n- Down-sampling frequency : %.2fHz"
            "\n- Number of time points (after down-sampling): %i"
            "\n- Number of channels : %i"
            )
    print(info % (binPath, sf, data.shape[1], len(channels)))


def read_TDT(binPath, downSample=None, tStart=None, tEnd=None, chanList=None,
             ds_method='interpolation'):
    """Load TDT data using the tdt python package.

    Args:
        binPath (str | pathlib.Path): Path to block

    Kwargs:
        downSample (int | float | None): Frequency in Hz at which the data is
            subsampled. No subsampling if None. (default None)
        tStart (float | None): Time in seconds from start of recording of first
            loaded sample. Default 0.0
        tEnd (float | None): Time in seconds from start of recording of last
            loaded sample. Duration of recording by default
        chanList (list(string)): List of loaded channels. Should be non empty.
            The provided list should be formatted as follows::
                    [<score_name>-<channel_index>, ...]
            Where channels are 1-indexed, (IMPORTANT) not 0-indexed (for
            consistency with tdt methods)
                eg: [LFPs-1, LFPs-2, EEGs-1, EEGs-94, EMGs-1...]
        ds_method (str): Method for resampling. Passed to
            ``resample.signal_resample``. 'poly' is more accurate,
            'interpolation' is faster (default 'interpolation')

    Returns:
        data (np.ndarray): The raw data of shape (n_channels, n_points)
        downsample (float):The down-sampling frequency used.
        chanList (list(str)): List of channels
    """
    import tdt

    print(f"Load TDT block at {binPath}")

    if tStart is None:
        tStart = 0.0
    if tEnd is None:
        tEnd = 0.0
    print(f"tStart = {tStart}, tEnd={tEnd}")

    def validate_chan(s):
        return isinstance(s, str) and len(s.split('-')) == 2

    def parse_chan(score_chan):
        store, chan = [s.strip(' ') for s in score_chan.split('-')]
        return store, int(chan)

    # Check length and formatting of `chanList parameter`
    chanList_error_msg = (
        "`chanList` should be a non-empty list of strings and formatted as "
        "follows:\n         [<score_name>-<channel_index>, ...], \n"
        "where channel indices are 1-indexed (not 0-indexed). eg: \n"
        "       [LFPs-1, LFPs-2, EEGs-1, EEGs-94, EMGs-1...] \n"
        f"Currently chanList = {chanList}"
    )
    if chanList is None:
        raise ValueError(chanList_error_msg)
    if not (len(chanList) > 0
            and all([validate_chan(s) for s in chanList])
            and all([parse_chan(s)[1] > 0 for s in chanList])):
        raise ValueError(chanList_error_msg)

    # Load and downsample data for all requested channels
    chan_dat_list = []  # List of channel data arrays to concatenate
    chan_ts_list = []  # List of timestamps for each channel
    # Iterate on channels:
    for store_chan in chanList:
        store, chan = parse_chan(store_chan)
        print(f"Load channel {chan} from store {store}", end=", ")
        blk = read_tdt_block(binPath, t1=tStart, t2=tEnd, store=store,
                             channel=chan)

        # Check that the requested data is actually there
        if store not in blk.streams.keys():
            stores = tdt.read_block(binPath, t2=1.0, nodata=True).streams.keys()
            raise Exception(f"Store `{store}` not found in data."
                            f" Existing stores = {stores}")

        sRate = blk.streams[store].fs
        chandat = blk.streams[store].data  # (nSamples x 0)-array

        # Downsample the data
        if downSample is None or downSample == sRate:
            downSample = sRate
            chan_dat_ds = chandat
        else:
            if downSample > sRate:
                print(
                    f"Warning: The resampling rate ({downSample}) is greater "
                    f"than the original sampling rate ({sRate})"
                )
            print(f"-> Resampling from {sRate}Hz to {downSample}Hz using "
                  f"'{ds_method}' method")
            if not chan_dat_list:
                # First channel: downsample to target
                chan_dat_ds = resample.signal_resample(
                    chandat, sampling_rate=sRate,
                    desired_sampling_rate=downSample,
                    method=ds_method,
                )
            else:
                # next channels: downsample to match first channel's length
                chan_dat_ds = resample.signal_resample(
                    chandat, sampling_rate=sRate,
                    desired_length=len(chan_dat_list[0]),
                    method=ds_method,
                )

        # Add data and timestamps
        chan_ts_ds = [blk.streams[store].start_time + i/downSample
                      for i in range(len(chan_dat_ds))]
        chan_dat_list.append(chan_dat_ds)
        chan_ts_list.append(chan_ts_ds)

    # Check same number of samples for all channels
    assert all(
        [dat.shape == chan_dat_list[0].shape for dat in chan_dat_list]
    )
    # Check data is aligned for all channels (same timestamps for each channel)
    assert all(
        [len(set(ts_list)) == 1 for ts_list in zip(*chan_ts_list)]
    )

    return np.stack(chan_dat_list), downSample, chanList


def read_tdt_block(binPath, t1=None, t2=None, store=None, channel=None):
    """Wrapper arount tdt.read_block that avoids bug in the function.

    tdt.read_block returns "channel 1 not found in store" if first channel
    of a single-channel store.
    """
    try:
        return tdt.read_block(binPath, t1=t1, t2=t2, store=store,
                              channel=channel)
    except Exception as e:
        # tdt.read_block returns "channel 1 not found in store" if first channel
        # of a single-channel store.
        single_chan_store = tdt.read_block(
                binPath, t2=1.0, nodata=True
            ).streams[store].data.ndim == 1  # data is 1-dim if there's one channel in store, 2 dim otherwise
        if single_chan_store and channel == 1:
            print(f"`{store}` have only 1 chan -> Contourn tdt.read_block bug")
            return tdt.read_block(
                binPath, t1=t1, t2=t2, store=store, channel=0
            )  # Return "all" channels of single channel store
        raise e


def read_SGLX(binPath, downSample=None, tStart=None, tEnd=None, chanList=None,
              chanListType='labels', ds_method='interpolation'):
    """Load SpikeGLX data.

    Args:
        binPath (str | pathlib.Path): Path to bin of recording

    Kwargs:
        downSample (int | float | None): Frequency in Hz at which the data is
            subsampled. No subsampling if None. (default None)
        tStart (float | None): Time in seconds from start of recording of first
            loaded sample. Default 0.0
        tEnd (float | None): Time in seconds from start of recording of last
            loaded sample. Duration of recording by default
        chanList (list(int) | None): List of loaded channels. All channels are
            loaded by default.
                eg: ["LF0;384", "LF1;385"]
        ChanListType (str): 'indices' or 'label'. If 'indices', chanList is
            interpreted as indices of saved channels. If 'labels', chanList is
            interpreted as labels of channels (eg: "LF0;384")
            not all channels are saved on file during a recording. (default
            'labels')
        ds_method (str): Method for resampling. Passed to
            ``resample.signal_resample``. 'poly' is more accurate,
            'interpolation' is faster (default 'interpolation')

    Returns:
        data (np.ndarray): The raw data of shape (n_channels, n_points)
        downsample (float):The down-sampling frequency used.
        channels (list(str)): List of channel names / original indices
    """
    print(f"Load SpikeGLX data at {binPath}")
    meta = SGLX.readMeta(Path(binPath))
    sRate = SGLX.SampRate(meta)

    # Indices in recording of first and last loaded samples
    if tStart is None:
        tStart = 0.0
    if tEnd is None:
        tEnd = float(meta['fileTimeSecs'])
    assert tStart <= tEnd

    # Indices of loaded channels in recording, and original labels
    assert chanList is None or chanList == 'all' or len(chanList) > 0, (
        "The chanList parameter should be None, 'all' or a non-empty list."
        f"Currently chanList = {chanList}"
    )
    savedLabels = SGLX.savedChanLabels(meta)
    chanIdxList, chanLblList = get_loaded_chans_idx_labels(
        chanList, chanListType, savedLabels
    )

    print(f"Loading N={len(chanIdxList)}/{len(savedLabels)} channels, "
          f"from tStart={tStart}s to tEnd={tEnd}s...")
    # Load RAW data
    DataRaw = SGLX.makeMemMapRaw(binPath, meta)[chanIdxList, :]

    # Convert raw data to requested unit
    # This transforms the memmap into nparray
    unit = 'uv'
    print(f"Convert data to {unit}")
    if unit.lower() == 'uv':
        factor = 1.e6
    if meta['typeThis'] == 'imec':
        # apply gain correction and convert
        convData = factor * SGLX.GainCorrectIM(DataRaw, chanIdxList, meta)
    else:
        # apply gain correction and convert
        convData = factor * SGLX.GainCorrectNI(DataRaw, chanIdxList, meta)

    # Downsample
    if downSample is None or downSample == sRate:
        downSample = sRate
        data_ds = convData
    else:
        if downSample > sRate:
            print(
                f"Warning: The resampling rate ({downSample}) is greater "
                f"than the original sampling rate ({sRate})"
            )
        print(f"-> Resampling from {sRate}Hz to {downSample}Hz using "
              f"'{ds_method}' method")
        ds_list = []
        for i in range(convData.shape[0]):
            if not ds_list:
                # First channel: downsample to target
                ds_list.append(
                    resample.signal_resample(
                        convData[i, :], sampling_rate=sRate,
                        desired_sampling_rate=downSample,
                        method=ds_method,
                    )
                )
            else:
                # next channels: downsample to match first channel's length
                ds_list.append(
                    resample.signal_resample(
                        convData[i, :], sampling_rate=sRate,
                        desired_length=len(ds_list[0]),
                        method=ds_method,
                    )
                )
        data_ds = np.vstack(ds_list)

    # Plot
    # import matplotlib.pyplot as plt
    # raw_ts = np.linspace(tStart, tEnd, convData.shape[1])
    # ds_ts = np.linspace(tStart, tEnd, data_ds.shape[1])
    # plt.plot(raw_ts, convData[0,:], 'r+-')
    # plt.plot(ds_ts, data_ds[0,:], 'b+-')
    # plt.show()

    return data_ds, downSample, chanLblList


def get_loaded_chans_idx_labels(chanList, chanListType, savedLabels):
    """Return lists of indices and labels of loaded channels."""
    if chanList is None or chanList == 'all':
        # Load all channels
        chanList = range(0, len(savedLabels))
        chanListType = 'indices'
    assert chanListType in ['indices', 'labels']
    if chanListType == 'indices':
        # Interpret the list of channels as a list of indices in saved recording
        chanIdxList = [int(c) for c in chanList]
        chanLblList = [savedLabels[i] for i in chanIdxList]
    elif chanListType == 'labels':
        errorchans = set(chanList) - set(savedLabels)
        if errorchans:
            raise Exception(
                f"The following channels labels were not found in data:"
                f"{errorchans}\n"
                f"Below are the labels of saved channels in the recording:\n"
                f"{savedLabels}"
            )
        # Interpret the list of channels as a list of labels
        chanIdxList, chanLblList = zip(
            *[
                (idx, lbl) for (idx, lbl) in enumerate(savedLabels)
                if lbl in chanList
            ]
        )
    return list(chanIdxList), list(chanLblList)


LOADING_FUNCTIONS = {
    'sglx': read_SGLX,
    'tdt': read_TDT,
}
