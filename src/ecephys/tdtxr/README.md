# TDT + xarray = TDTxr
Utilities for loading TDT data in xarray format.

## Example usage
```
>>> import ecephys.tdtxr as txr
>>> from pathlib import Path
>>> block_path = Path('/path/to/block')

>>> store_names = txr.load_store_names(block_path)
>>> channel_lists = txr.load_channel_lists(block_path)
>>> print(f"Store names: {store_names}")
>>> print(f"Channel lists: {channel_lists}")

Store names: ['EEG_', 'EEGr', 'EMGr']
Channel lists: {'EEG_': [1, 2, 3, 4, 5, 6], 'EEGr': [1, 2, 3, 4, 5, 6], 'EMGr': [1, 2]}

>>> store_name = 'EEGr'
>>> channels = [1, 3, 6]
>>> start_time, end_time = (0, 60)

>>> sig = txr.load_stream_store(block_path, store_name, chans=channels, start_time=start_time, end_time=end_time)
>>> print(sig)

<xarray.DataArray 'EEGr' (time: 91553, channel: 3)>
array([[  3.136    ,  57.407997 ,  24.703999 ],
       [ 19.904    ,  59.519997 ,  16.64     ],
       [ 11.903999 ,  57.024    ,  15.487998 ],
       ...,
       [  4.8639994, -24.895998 ,   5.568    ],
       [  6.8479996, -22.463999 ,   5.7599998],
       [ 20.735998 ,  -7.3599997,   2.4319997]], dtype=float32)
Coordinates:
  * time       (time) float64 0.0 0.0006554 0.001311 0.001966 ... 60.0 60.0 60.0
  * channel    (channel) int64 1 3 6
    timedelta  (time) timedelta64[ns] 00:00:00 ... 00:00:59.999518720
    datetime   (time) datetime64[ns] 2021-04-21T09:08:28.999999 ... 2021-04-2...
Attributes:
    units:    uV
    fs:       1525.87890625
```