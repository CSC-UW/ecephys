import numpy as np
import pandas as pd
import tdt
import xarray as xr
import yaml
from pathlib import Path
import hypnogram as hp
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import ecephys.plot as eplt
import ecephys.xrsig as xrsig
from scipy.stats import mode
from ripple_detection.core import gaussian_smooth


##Functions for loading TDT SEV-stores, and visbrain hypnograms:
def load_hypnograms(subject, experiment, condition, scoring_start_time):
    hypnograms_yaml_file = "N:\Data\paxilline_project_materials\pax-hypno-paths.yaml"

    with open(hypnograms_yaml_file) as fp:
        yaml_data = yaml.safe_load(fp)

    root = Path(yaml_data[subject]["hypno-root"])
    hypnogram_fnames = yaml_data[subject][experiment][condition]
    hypnogram_paths = [root / (fname + ".txt") for fname in hypnogram_fnames]

    hypnogram_start_times = pd.date_range(
        start=scoring_start_time, periods=len(hypnogram_paths), freq="7200S"
    )
    hypnograms = [
        hp.load_visbrain_hypnogram(path).as_datetime(start_time)
        for path, start_time in zip(hypnogram_paths, hypnogram_start_times)
    ]

    return pd.concat(hypnograms).reset_index(drop=True)

def sev_to_xarray(info, store):
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
    n_channels, n_samples = store.data.shape

    time = np.arange(0, n_samples) / store.fs + store.start_time
    timedelta = pd.to_timedelta(time, "s")
    datetime = pd.to_datetime(info.start_date) + timedelta

    volts_to_microvolts = 1e6
    data = xr.DataArray(
        store.data.T * volts_to_microvolts,
        dims=("time", "channel"),
        coords={
            "time": time,
            "channel": store.channels,
            "timedelta": ("time", timedelta),
            "datetime": ("time", datetime),
        },
        name=store.name,
    )
    data.attrs["units"] = "uV"
    data.attrs["fs"] = store.fs

    return data

def load_sev_store(path, t1=0, t2=0, channel=None, store=''):

    data = tdt.read_block(path, channel=channel, store=store, t1=t1, t2=t2)
    store = data.streams[store]
    info = data.info
    datax = sev_to_xarray(info, store)
    return datax

#Functions used for working with xset-style dictionaries which contain all relevant information for a given experiment
def get_key_list(dict):
    list = []
    for key in dict.keys():
        list.append(key) 
    return list

def save_xset(ds, analysis_root):
    """saves each component of an experimental 
    dataset dictionary (i.e. xr.arrays of the raw data and of the spectrograms), 
    as its own separate .nc file. All can be loaded back in as an experimental dataset dictionary
    using fetch_xset
    """
    keys = get_key_list(ds)
    for key in keys:
        path = analysis_root / (ds['name'] + key + ".nc") 
        ds[key].to_netcdf(path)
    print('Remember to save key list in order to fetch the data again')

def fetch_xset(exp, key_list, analysis_root):
    #exp is a string, key list is a list of strings
    dataset = {}
    dataset['name'] = exp
    for key in key_list: 
        path = analysis_root / (exp + key + ".nc")
        try:
            dataset[key] = xr.load_dataarray(path)
        except: 
            dataset[key] = xr.load_dataset(path)
    return dataset

def get_data_spg(block_path, store='', t1=0, t2=0, channel=None):
    data = kd.load_sev_store(block_path, t1=t1, t2=t2, channel=channel, store=store)
    spg = kd.get_spextrogram(data)
    print('Remember to save all data in xset-style dictionary, and to add experiment name key (key = "name") before using save_xset')
    return data, spg


## Spectrogram Utils
def get_spextrogram(sig, window_length=4, overlap=1, **kwargs):
    kwargs['nperseg'] = int(window_length * sig.fs) # window length in number of samples
    kwargs['noverlap'] = int(overlap * sig.fs) # overlap in number of samples
    spg = xrsig.parallel_spectrogram_welch(sig, **kwargs)
    return spg

def get_bp_set(spg, bands):
    if type(spg) == xr.core.dataset.Dataset:
        spg = spg.to_array(dim='channel')
    
    bp_ds = xr.Dataset(
    {
        "delta": get_bandpower(spg, bands['delta']),
        "theta": get_bandpower(spg, bands['theta']),
        "beta": get_bandpower(spg, bands['beta']),
        "low_gamma": get_bandpower(spg, bands['low_gamma']),
        "high_gamma": get_bandpower(spg, bands['high_gamma']),
    })
    return bp_ds

def get_ss_spg(spg, hypno, states, bands, t1=None, t2=None):
    """ returns a bandpower dataset where all timechunks and corresponding datapoints are dropped"""
    if type(spg) == xr.core.dataarray.DataArray:
        spg = spg.to_dataset(dim='channel')
    ss_spg = filter_dataset_by_state(spg, hypno, states)
    ss_da = ss_spg.to_array(dim='channel')
    bp_ds = xr.Dataset(
    {
        "delta": get_bandpower(ss_da, bands['delta']),
        "theta": get_bandpower(ss_da, bands['theta']),
        "beta": get_bandpower(ss_da, bands['beta']),
        "low_gamma": get_bandpower(ss_da, bands['low_gamma']),
        "high_gamma": get_bandpower(ss_da, bands['high_gamma']),
    })
    if t2 is not None: 
        bp_ds = bp_ds.isel(time=slice(t1, t2))
    return ss_da, bp_ds

def get_bandpower(spg, f_range):
    """Get band-limited power from a spectrogram.
    Parameters
    ----------
    spg: xr.DataArray (frequency, time, [channel])
        Spectrogram data.
    f_range: (float, float)
        Frequency range to restrict to, as [f_low, f_high].
    Returns:
    --------
    bandpower: xr.DataArray (time, [channel])
        Sum of the power in `f_range` at each point in time.
    """
    bandpower = spg.sel(frequency=slice(*f_range)).sum(dim="frequency")
    bandpower.attrs["f_range"] = f_range

    return bandpower


#Misc utils for dealing with xarray structures: 
def estimate_fs(da):
    sample_period = mode(np.diff(da.datetime.values)).mode[0]
    assert isinstance(sample_period, np.timedelta64)
    sample_period = sample_period / pd.to_timedelta(1, "s")
    return 1 / sample_period

def get_smoothed_da(da, smoothing_sigma=10, in_place=False):
    if not in_place:
        da = da.copy()
    da.values = gaussian_smooth(da, smoothing_sigma, estimate_fs(da))
    return da

def get_smoothed_ds(ds, smoothing_sigma=10, in_place=False):
    if not in_place:
        ds = ds.copy()
    for da_name, da in ds.items():
        ds[da_name] = get_smoothed_da(da, smoothing_sigma, in_place)
    return ds