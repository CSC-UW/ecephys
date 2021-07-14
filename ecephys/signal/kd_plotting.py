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


#stole this function from Graham's 'compare_bandpower_timeseries' notebook. Any changes KD makes are noted with comments.
state_colors = {
    "Wake": "forestgreen",
    "Brief-Arousal": "chartreuse",
    "Transition-to-NREM": "lightskyblue",
    "Transition-to-Wake": "palegreen",
    "NREM": "royalblue",
    "Transition-to-REM": "plum",
    "REM": "magenta",
    "Transition": "grey",
    "Art": "crimson",
    "Unsure": "white",
}

def _plot_spectrogram_with_bandpower(spg, bp, hyp, title=None, figsize=(15, 5)):
    fig, (bp_ax, spg_ax) = plt.subplots(
        ncols=1,
        nrows=2,
        figsize=figsize,
        gridspec_kw=dict(width_ratios=[1], height_ratios=[1, 1]),
    )

    bp = ux.get_smoothed_da(bp)
    sns.lineplot(x=bp.datetime, y=bp, color="black", ax=bp_ax)
    bp_ax.set(xlabel=None, ylim=(0,350), ylabel='Percent of BL NREM Delta Mean', xticks=[], xmargin=0)
    eplt.plot_hypnogram_overlay(
        hyp, xlim=bp_ax.get_xlim(), state_colors=state_colors, ax=bp_ax
    )

    eplt.plot_spectrogram(
        spg.frequency, spg.datetime, spg, yscale="log", f_range=(0, 50), ax=spg_ax
    )

    if title:
        fig.suptitle(title)
    plt.tight_layout(h_pad=0.0)
    return bp_ax, spg_ax


def plot_spectrogram_with_bandpower(spg, bp, hyp, channel, start_time, end_time, title=None):
    start = hyp.start_time.min() + pd.to_timedelta(start_time)
    end = hyp.start_time.min() + pd.to_timedelta(end_time)
    b, s = _plot_spectrogram_with_bandpower(
        spg.sel(channel=channel, time=slice(start_time, end_time)),
        bp.sel(channel=channel, time=slice(start_time, end_time)),
        hyp,
        title=title,
    )
    return b, s


def plot_main_pax(s, p, sbl, pbl, band='delta', t1=None, t2=None):
    """Takes state-specfic bandpower sets"""
    if t2 is not None: 
        s = s.isel(time=slice(t1, t2))
        p = p.isel(time=slice(t1, t2))

    #weird that I couldn't accomplish this step with a list... there must be a way to do it
    s = s[[band]].to_array(name=band).squeeze('variable', drop=True)
    p = p[[band]].to_array(name=band).squeeze('variable', drop=True)
    sbl = sbl[[band]].to_array(name=band).squeeze('variable', drop=True)
    pbl = pbl[[band]].to_array(name=band).squeeze('variable', drop=True)
    
    sr2bl = (s / sbl.mean(dim='time')) * 100
    pr2bl = (p / pbl.mean(dim='time')) * 100
    sdf = sr2bl.to_dataframe().assign(condition='Saline')
    pdf = pr2bl.to_dataframe().assign(condition='Paxilline')
    sdf.reset_index(inplace=True)
    pdf.reset_index(inplace=True)
    
    sp = pd.concat([sdf, pdf])
    
    f, ax = plt.subplots(figsize=(9,7))
    sns.boxplot(x="channel", y=band,
            hue="condition", palette=["plum", "gold"],
            data=sp)
    sns.despine(offset=10, trim=True)
    ax.set_xlabel('Channel')
    ax.set_ylabel(band.capitalize() + ' Power as % of BL Mean')
    #ax.set_title('PAX_4 Experiment-2, Full (4-Hr) NREM Delta Rebound as % of BL Mean')
    return ax


def plot_swa_r2bl(s, sbl, p, pbl, channel=1, ss=12, title = ''):
    """Takes state-specfifc bandpower sets"""
    s=s.sel(channel=channel)
    p=p.sel(channel=channel)
    sbl=sbl.sel(channel=channel)
    pbl=pbl.sel(channel=channel)
    srel = s.delta / sbl.delta.mean(dim='time') * 100
    srels = ux.get_smoothed_da(srel, smoothing_sigma=ss)
    prel = p.delta / pbl.delta.mean(dim='time') * 100
    prels = ux.get_smoothed_da(prel, smoothing_sigma=ss)
    sal = srels.to_dataframe().assign(condition='Saline')
    sal.reset_index(inplace=True)
    pax = prels.to_dataframe().assign(condition='Paxilline')
    pax.reset_index(inplace=True)
    
    pax['rel_time'] = np.arange(0, len(pax), 1)
    sal['rel_time'] = np.arange(0, len(sal), 1)
    
    sp = pd.concat([sal, pax])
    
    f, ax = plt.subplots(figsize=(18, 6))
    sns.lineplot(x='rel_time', y='delta', hue='condition', data=sp, palette=["darkorchid", "goldenrod"], dashes=False, ax=ax)
    ax.set_title(title)
    ax.set_ylabel('Delta Power (0.5-4Hz) as % of Baseline')
    ax.set_xlabel("Time (ignore the raw values)")
    return ax