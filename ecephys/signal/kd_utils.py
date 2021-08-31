import numpy as np
import pandas as pd
import tdt
import xarray as xr
import yaml
from pathlib import Path
import hypnogram as hp
import ecephys.signal.timefrequency as tfr
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import ecephys.plot as eplt
import ecephys.utils.xarray as ux
import ecephys.signal.xarray_utils as ecx


def load_hypnos(yaml_path, subject, experiment, condition, ext='.txt', scoring_start_time=None):
    with open(yaml_path) as fp:
        yaml_data = yaml.safe_load(fp)
    root = Path(yaml_data[subject]['hypno-root'])
    hypnos = yaml_data[subject][experiment][condition]
    
    
    hypnogram_paths = []
    for hyp in hypnos:
        hyp = hyp+ext
        ppath = root / hyp
        hypnogram_paths += [ppath]
    
    hypnograms = [hp.load_visbrain_hypnogram(path) for path in hypnogram_paths]
    hypnogram_duration = 7200.0
    hypnogram_offsets = np.arange(0, len(hypnograms)) * hypnogram_duration
    
    for hypnogram, offset in zip(hypnograms, hypnogram_offsets):
        hypnogram['start_time'] = pd.to_timedelta(hypnogram['start_time'] + offset, 's')
        hypnogram['end_time'] = pd.to_timedelta(hypnogram['end_time'] + offset, 's')
        hypnogram['duration'] = pd.to_timedelta(hypnogram['duration'], 's')
        
    hypnogram = pd.concat(hypnograms).reset_index(drop=True)
    
    if scoring_start_time:
        hypnogram['start_time'] = hypnogram['start_time'] + scoring_start_time
        hypnogram['end_time'] = hypnogram['end_time'] + scoring_start_time
    
    return hp.DatetimeHypnogram(hypnogram)

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


def fetch_data(data_type='spg', spg_array=True, path=''):

    """ Loads spectrogram, hypnograms, or bandpower datasets
    """

    if spg_array==True: 
        spg_array = xr.load_dataarray(path)
        return spg_array
    elif data_type=='spg': 
        spg = xr.load_dataarray(path).to_dataset('channel')
        return spg
    elif data_type=='hyp':
        hyp = hp.load_datetime_hypnogram(path)
        return hyp
    elif data_type=='bp':
        bp = xr.load_dataset(path)
        return bp
    else: 
        return print('Choose a valid data type - spg, hyp, or bp')


def get_spextrogram(sig, window_length=4, overlap=1, **kwargs):
    kwargs['nperseg'] = int(window_length * sig.fs) # window length in number of samples
    kwargs['noverlap'] = int(overlap * sig.fs) # overlap in number of samples
    spg = tfr.parallel_spectrogram_welch(sig, **kwargs)
    return spg


def get_bp_set(spg, bands):
    if type(spg) == xr.core.dataset.Dataset:
        spg = spg.to_array(dim='channel')
    
    bp_ds = xr.Dataset(
    {
        "delta": tfr.get_bandpower(spg, bands['delta']),
        "theta": tfr.get_bandpower(spg, bands['theta']),
        "beta": tfr.get_bandpower(spg, bands['beta']),
        "low_gamma": tfr.get_bandpower(spg, bands['low_gamma']),
        "high_gamma": tfr.get_bandpower(spg, bands['high_gamma']),
    }
)
    return bp_ds



def get_ss_bp_set(spg, hypno, states, bands, t1=None, t2=None):
    """ returns a bandpower dataset where all timechunks and corresponding datapoints are dropped"""
    if type(spg) == xr.core.dataarray.DataArray:
        spg = spg.to_dataset(dim='channel')
    ss_spg = ecx.filter_dataset_by_state(spg, hypno, states)
    ss_da = ss_spg.to_array(dim='channel')
    bp_ds = xr.Dataset(
    {
        "delta": tfr.get_bandpower(ss_da, bands['delta']),
        "theta": tfr.get_bandpower(ss_da, bands['theta']),
        "beta": tfr.get_bandpower(ss_da, bands['beta']),
        "low_gamma": tfr.get_bandpower(ss_da, bands['low_gamma']),
        "high_gamma": tfr.get_bandpower(ss_da, bands['high_gamma']),
    }
)
    if t2 is not None: 
        bp_ds = bp_ds.isel(time=slice(t1, t2))
    return bp_ds

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