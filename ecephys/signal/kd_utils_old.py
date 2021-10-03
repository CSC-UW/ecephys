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

"""Retired from kd_utils Module:"""

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

def fetch_data(data_type='spg', spg_array=False, path=''):

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

def get_ss_spg(spg, hypno, states, bands, t1=None, t2=None):
    """ returns a spectrogram corresponding only to the desired states specified by a matching hypnogram """
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