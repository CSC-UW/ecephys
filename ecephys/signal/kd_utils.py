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
import ecephys.xrsig.hypnogram_utils as xrhyp
import ecephys.signal.kd_plotting as kp
from scipy.stats import mode
from ripple_detection.core import gaussian_smooth
import tdt_xarray as tx

##Functions for loading TDT SEV-stores, and visbrain hypnograms:
def load_hypnograms(subject, experiment, condition, scoring_start_time, hypnograms_yaml_file="/Volumes/paxilline/Data/paxilline_project_materials/pax-hypno-paths.yaml"):
    
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
    # had to add this try-except because stupid TDT defines 'channels' for EEG/LFP, but 'channel' for EMG. 
    try:
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
    except:
        data = xr.DataArray(
            store.data.T * volts_to_microvolts,
            dims=("time", "channel"),
            coords={
                "time": time,
                "channel": store.channel,
                "timedelta": ("time", timedelta),
                "datetime": ("time", datetime),
            },
            name=store.name,
        )
    data.attrs["units"] = "uV"
    data.attrs["fs"] = store.fs

    return data

def tev_to_xarray(info, store):
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
            "channel": store.channel,
            "timedelta": ("time", timedelta),
            "datetime": ("time", datetime),
        },
        name=store.name,
    )
    data.attrs["units"] = "uV"
    data.attrs["fs"] = store.fs

    return data

def load_tev_store(path, t1=0, t2=0, channel=None, store=''):

    data = tdt.read_block(path, channel=channel, store=store, t1=t1, t2=t2)
    store = data.streams[store]
    info = data.info
    datax = tev_to_xarray(info, store)
    return datax

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

def save_key_list(key_list, text_file='key_lists.txt'):
    with open(text_file, "w") as output:
        output.write(str(key_list))

def save_xset(ds, analysis_root):
    """saves each component of an experimental 
    dataset dictionary (i.e. xr.arrays of the raw data and of the spectrograms), 
    as its own separate .nc file. All can be loaded back in as an experimental dataset dictionary
    using fetch_xset
    """
    keys = get_key_list(ds)
    for key in keys:
        try:
            path = analysis_root / (ds['name'] + "_" + key + ".nc") 
            ds[key].to_netcdf(path)
        except AttributeError:
            path = analysis_root / (ds['name'] + "_" + key + ".tsv") 
            ds[key].write(path)
        except:
            pass


    print('Remember to save key list in order to fetch the data again')

def fetch_xset(exp, key_list, analysis_root):
    #exp is a string, key list is a list of strings
    dataset = {}
    for key in key_list: 
        try:
            path = analysis_root / (exp + "_" + key + ".nc")
            dataset[key] = xr.load_dataarray(path)
        except:
            path = analysis_root / (exp + "_" + key + ".tsv")
            dataset[key] = hp.load_datetime_hypnogram(path) 
    dataset['name'] = exp
    return dataset

def get_data_spg(block_path, store='', t1=0, t2=0, channel=None, sev=True, window_length=4, overlap=1, pandas=False):
    if sev == True:
        data = load_sev_store(block_path, t1=t1, t2=t2, channel=channel, store=store)
    else:
        data = load_tev_store(block_path, t1=t1, t2=t2, channel=channel, store=store)
    spg = get_spextrogram(data, window_length=window_length, overlap=overlap)
    print('Remember to save all data in xset-style dictionary, and to add experiment name key (key = "name") before using save_xset')
    data = data.swap_dims({'time': 'datetime'})
    spg = spg.swap_dims({'time': 'datetime'})
    if pandas == True: 
        data = data.to_dataframe().drop(labels=['time', 'timedelta'], axis=1)
        spg = spg.to_dataframe(name='Power').drop(labels=['time', 'timedelta'], axis=1)
    return data, spg


## Spectrogram Utils
def get_spextrogram(sig, window_length=4, overlap=1, **kwargs):
    kwargs['nperseg'] = int(window_length * sig.fs) # window length in number of samples
    kwargs['noverlap'] = int(overlap * sig.fs) # overlap in number of samples
    spg = xrsig.parallel_spectrogram_welch(sig, **kwargs)
    return spg

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

def get_pob_bp(bl_spg, exp_spg, bl_hyp, f_range=(0.5, 4), states=['NREM']):
    #get bandpower for both exp and bl
    exp_bp = get_bandpower(exp_spg, f_range) 
    bl_bp = get_bandpower(bl_spg, f_range)
    #make the bl bandpower specific for a given set of states
    bl_ss_bp = xrhyp.keep_states(bl_bp, bl_hyp, states)
    #average the state-specific bl bandpower over time
    bl_ss_bp_avg = bl_ss_bp.mean(dim='time')
    #divide experimental bandpower by bl bandpower
    exp_bp_ponb = exp_bp/bl_ss_bp_avg
    return exp_bp_ponb*100

def get_bp_set(spg, bands):
    if type(spg) == xr.core.dataset.Dataset:
        spg = spg.to_array(dim='channel')
    
    bp_ds = xr.Dataset(
    {
        "delta": get_bandpower(spg, bands['delta']),
        "theta": get_bandpower(spg, bands['theta']),
        "sigma": get_bandpower(spg, bands['sigma']),
        "beta": get_bandpower(spg, bands['beta']),
        "low_gamma": get_bandpower(spg, bands['low_gamma']),
        "high_gamma": get_bandpower(spg, bands['high_gamma']),
    })
    return bp_ds

def get_bp_set2(spg, bands, pandas=False):
    if type(spg) == xr.core.dataset.Dataset:
        spg = spg.to_array(dim='channel')
    
    bp_ds = xr.Dataset({})
    bp_vars = {}
    keys = get_key_list(bands)
    for k in keys:
        bp_vars[k] = get_bandpower(spg, bands[k])
    bp_set = bp_ds.assign(**bp_vars)
    if pandas == True:
        bp_set = bp_set.to_dataframe().drop(labels=['time', 'timedelta'], axis=1)
        return bp_set
    else:
        return bp_set

def get_ss_spg_bp(spg, hyp, state, bands):
    """Need datetime dimension"""
    try:
        spg = spg.swap_dims({'time': 'datetime'})
    except:
        pass
    spg = spg.sel(datetime=hyp.keep_states(state).covers_time(spg.datetime)).dropna(dim='datetime')
    bp = get_bp_set(spg, bands)
    return spg.swap_dims({'datetime': 'time'}), bp.swap_dims({'datetime': 'time'})

def get_ss_bp(spg, hyp, state, bands):
    """Need datetime dimension"""
    try:
        spg = spg.swap_dims({'time': 'datetime'})
    except:
        pass
    spg = spg.sel(datetime=hyp.keep_states(state).covers_time(spg.datetime)).dropna(dim='datetime')
    bp = get_bp_set(spg, bands)
    return bp.swap_dims({'datetime': 'time'})


def get_ss_spg(spg, hyp, state, dt=True):
    """Need datetime dimension"""
    try:
        spg = spg.swap_dims({'time': 'datetime'})
    except:
        pass
    spg = spg.sel(datetime=hyp.keep_states(state).covers_time(spg.datetime)).dropna(dim='datetime')
    if dt == True:
        return spg
    else:
        return spg.swap_dims({'datetime': 'time'})


def get_ss_psd(spg, hyp, state, median=True):
    if median==True:
        return get_ss_spg(spg, hyp, state).median(dim="datetime")
    else: 
        return get_ss_spg(spg, hyp, state).mean(dim="datetime")

def compare_spectra(spg1, spg2, hyp1, hyp2, channel, keys):
    
    def compare_state_psd(state, scale='log', freqs=slice(None), channel=channel, keys=keys):
        psd1 = xrhyp.keep_states(spg1, hyp1, state).sel(frequency=freqs, channel=channel)
        psd2 = xrhyp.keep_states(spg2, hyp2, state).sel(frequency=freqs, channel=channel)
        return kp.compare_psd(psd1, psd2, state=state, keys=keys, scale=scale)
    
    g = compare_state_psd(['NREM'])
    g = compare_state_psd(['Wake'])

    g = compare_state_psd(['NREM'], scale='linear', freqs=slice(0, 40))
    g = compare_state_psd(['Wake'], scale='linear', freqs=slice(0, 40))

def n_freq_bins(da, f_range):
    return da.sel(frequency=slice(*f_range)).frequency.size

def get_psd_rel2bl(bl_spg, exp_spg, bl_hyp, exp_hyp, state, chan, median=True):
    bl_psd = get_ss_psd(bl_spg, bl_hyp, state, median=median)
    exp_psd = get_ss_psd(exp_spg, exp_hyp, state, median=median)
    return (exp_psd / bl_psd *100).sel(channel=chan)

def get_auc(psd, f_range):
        """ Sums all power (all 'area under curve') in a given f_range of the PSD, and divides by the number of frequecy bins  
        psd: xr.data_array. Could use kd.get_ss_psd for example"""
        return psd.sel(frequency=slice(*f_range)).sum(dim='frequency') / n_freq_bins(psd, f_range)

def compare_auc(psd1, psd2, f_range, title='Relative SWA Rebound (PAX-SAL), NREM-Only'):
    """Uses get_auc on two separate PSD's, then compares them by taking PSD2-PSD1"""
    psd1 = get_auc(psd1, f_range)
    psd2 = get_auc(psd2, f_range)
    comp = psd2 - psd1
    return comp.to_dataframe(title)

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

#Misc Analysis Utils:
def get_frac_oc(hyps, hyp_keys):
    df = pd.concat([hyp.fractional_occupancy().to_frame() for hyp in hyps], keys=hyp_keys).unstack()*100
    return df