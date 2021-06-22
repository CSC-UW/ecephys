import numpy as np
import pandas as pd
import tdt
import xarray as xr

def load_tdt_xarray(blk_path, store=None, channel=None, t1=0, t2=0):
    kwargs = dict(store=store, channel=channel, t1=t1, t2=t2)
    blk = tdt.read_block(blk_path, **kwargs)
    stream = blk.streams[store]
    
    time = np.arange(0, stream.data.shape[1]) / stream.fs
    timedelta = pd.to_timedelta(time, "s")
    datetime = pd.to_datetime(blk.info.start_date) + timedelta
    
    volts_to_microvolts = 1e6
    
    # Wrap data with Xarray
    data = xr.DataArray(
        stream.data.T * volts_to_microvolts, 
        dims=("time", "channel"), 
        coords={
            "time": time, 
            "channel": stream.channel, 
            "timedelta": ("time", timedelta), 
            "datetime": ("time", datetime)
        }
    )
    
    data.attrs["units"] = 'uV'
    data.attrs["fs"] = stream.fs
    
    return data