import numpy as np
import pandas as pd
import tdt
import xarray as xr
import yaml
from pathlib import Path
import hypnogram as hp
import ecephys.signal.timefrequency as tfr
import xarray as xr

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
    
    return hypnogram

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

def get_spextrogram(sig, window_length=4, overlap=1, **kwargs):
    kwargs['nperseg'] = int(window_length * sig.fs) # window length in number of samples
    kwargs['noverlap'] = int(overlap * sig.fs) # overlap in number of samples
    spg = tfr.parallel_spectrogram_welch(sig, **kwargs)
    return spg