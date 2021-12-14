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
import ecephys.xrsig as xrsig
import ecephys.signal.kd_utils as kd
import neurodsp.plts.utils as dspu
import neurodsp.spectral as dsps



hypno_colors = {
    "Wake": "forestgreen",
    "Brief-Arousal": "chartreuse",
    "Transition-to-NREM": "lightskyblue",
    "Transition-to-Wake": "palegreen",
    "NREM": "royalblue",
    "Transition-to-REM": "plum",
    "REM": "magenta",
    "Transition": "grey",
    "Art": "crimson",
    "Wake-art": "crimson",
    "Unsure": "white",
}

def quick_lineplot(data):
    f, ax = plt.subplots(figsize=(35, 10))
    ax = sns.lineplot(x=data.datetime, y=data.values, ax=ax)
    return ax

def shade_hypno_for_me(
    hypnogram, ax=None, xlim=None
):
    """Shade plot background using hypnogram state.

    Parameters
    ----------
    hypnogram: pandas.DataFrame
        Hypnogram with with state, start_time, end_time columns.
    ax: matplotlib.Axes, optional
        An axes upon which to plot.
    """
    xlim = ax.get_xlim() if (ax and not xlim) else xlim

    ax = dspu.check_ax(ax)
    for bout in hypnogram.itertuples():
        ax.axvspan(
            bout.start_time,
            bout.end_time,
            alpha=0.3,
            color=hypno_colors[bout.state],
            zorder=1000,
            ec="none",
        )

    ax.set_xlim(xlim)
    return ax

def plot_shaded_bp(spg, chan, bp_def, band, hyp, ax):
    bp_set = kd.get_bp_set2(spg, bp_def)
    
    bp = bp_set[band].sel(channel=chan)    
    bp = kd.get_smoothed_da(bp, smoothing_sigma=14)
    
    ax = sns.lineplot(x=bp.datetime, y=bp, ax=ax)
    shade_hypno_for_me(hypnogram=hyp, ax=ax)

    ax.set(xlabel=None, ylabel='Raw '+band.capitalize()+' Power', xticks=[], xmargin=0)
    return ax

def spectro_plotter(
    spg,
    chan,
    f_range=slice(0, 50),
    t_range=None,
    yscale="linear",
    figsize=(35, 10),
    vmin=None,
    vmax=None,
    title = 'Title',
    ax=None,
    ):
    try:
        #spg = spg.swap_dims({'datetime': 'time'})
        spg = spg.sel(channel=chan, frequency=f_range)
    except IndexError:
        print('Already had time dimension - passing index error')
    

    freqs = spg.frequency
    spg_times = spg.datetime.values
    #freqs, spg_times, spg = dsps.trim_spectrogram(freqs, spg_times, spg, f_range, t_range)

    ax = dspu.check_ax(ax, figsize=figsize)
    im = ax.pcolormesh(spg_times, freqs, np.log10(spg), cmap='nipy_spectral', vmin=vmin, vmax=vmax, alpha=0.5, shading="gouraud")
    #ax.figure.colorbar(im)
    ax.set_yscale(yscale)
    ax.set_ylabel("Frequency [Hz]")
    ax.set_xlabel("Time [sec]")
    ax.set_title(title)

    if yscale == "log":
        ax.set_ylim(np.min(freqs[freqs > 0]), np.max(freqs))
    return ax

def plot_bp_and_spectro(spg, chan, hyp, bp_def, band):
    f, (bx, sx) = plt.subplots(nrows=2, ncols=1, figsize=(35, 10), sharex=True)
    bx = plot_shaded_bp(spg, chan, bp_def, band, hyp, ax=bx)
    sx = spectro_plotter(spg, chan, ax=sx)
    return bx, sx



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


def plot_swa_r2bl(s, p, sbl, pbl, band, channel=1, ss=12, title = ''):
    """Takes state-specfifc bandpower sets"""
    s=s.sel(channel=channel)
    p=p.sel(channel=channel)
    sbl=sbl.sel(channel=channel)
    pbl=pbl.sel(channel=channel)
    srel = s[band] / sbl[band].mean(dim='time') * 100
    srels = kd.get_smoothed_da(srel, smoothing_sigma=ss)
    prel = p[band] / pbl[band].mean(dim='time') * 100
    prels = kd.get_smoothed_da(prel, smoothing_sigma=ss)
    sal = srels.to_dataframe().assign(condition='Saline')
    sal.reset_index(inplace=True)
    pax = prels.to_dataframe().assign(condition='Paxilline')
    pax.reset_index(inplace=True)
    
    pax['rel_time'] = np.arange(0, len(pax), 1)
    sal['rel_time'] = np.arange(0, len(sal), 1)
    
    sp = pd.concat([sal, pax])
    
    f, ax = plt.subplots(figsize=(18, 6))
    sns.lineplot(x='rel_time', y=band, hue='condition', data=sp, palette=["darkorchid", "goldenrod"], dashes=False, ax=ax)
    ax.set_title(title)
    ax.set_ylabel(band+' Power as % of Baseline')
    ax.set_xlabel("Time (ignore the raw values)")
    return ax


def compare_psd(
    psd1, psd2, state, keys=["condition1", "condition2"], key_name="condition", scale="log"
):
    df = pd.concat(
        [psd1.to_dataframe("power"), psd2.to_dataframe("power")], keys=keys
    ).rename_axis(index={None: key_name})
    g = sns.relplot(
        data=df,
        x="frequency",
        y="power",
        hue=key_name,
        col="channel",
        kind="line",
        aspect=(16 / 9),
        height=3,
        ci=None,
    )
    g.set(xscale=scale, yscale=scale, ylabel='Power, '+state[0]+' PSD')
    return g


def plot_bp_set(spg, bands, hyp, channel, start_time, end_time, ss=12, figsize=(14,7), title=None):
    spg = spg.sel(channel=channel, datetime=slice(start_time, end_time))
    bp_set = kd.get_bp_set2(spg, bands)
    bp_set = kd.get_smoothed_ds(bp_set, smoothing_sigma=ss)
    ax_index = np.arange(0, len(bands))
    keys = kd.get_key_list(bands)

    fig, axes = plt.subplots(ncols=1, nrows=len(bands), figsize=figsize)

    for i, k in zip(ax_index, keys):
        fr = bp_set[k].f_range
        fr_str = '('+str(fr[0]) + ' -> ' +str(fr[1])+' Hz)'
        ax = sns.lineplot(x=bp_set[k].datetime, y=bp_set[k], ax=axes[i])
        ax.set_ylabel('Raw '+k.capitalize()+' Power')
        ax.set_title(k.capitalize()+' Bandpower '+fr_str)
    fig.suptitle(title)
    fig.tight_layout(pad=0.5)
    return fig, axes




## OLD FUNCTIONS BORDERLINE WORTH KEEPING:

def _plot_spectrogram_with_bp_rel2bl(spg_exp, spg_bl, band, bp_def, hyp, title=None, figsize=(15, 5)):
    fig, (bp_ax, spg_ax) = plt.subplots(
        ncols=1,
        nrows=2,
        figsize=figsize,
        gridspec_kw=dict(width_ratios=[1], height_ratios=[1, 1]),
    )

    exp_bp = kd.get_bp_set(spg_exp, bp_def)
    bl_bp = kd.get_bp_set(spg_bl, bp_def)
    exp_pob = (exp_bp[band]/bl_bp[band].median(dim='time'))*100
    exp_pob = kd.get_smoothed_da(exp_pob, smoothing_sigma=12)
    sns.lineplot(x=exp_pob.datetime, y=exp_pob, color="black", ax=bp_ax)
    bp_ax.set(xlabel=None, ylabel=band.capitalize()+' Power as % of BL', xticks=[], xmargin=0)
    if hyp is not None:
        eplt.plot_hypnogram_overlay(
            hyp, xlim=bp_ax.get_xlim(), state_colors=state_colors, ax=bp_ax
        )

    eplt.plot_spectrogram(
        spg_exp.frequency, spg_exp.datetime, spg_exp, yscale="log", f_range=(0, 50), ax=spg_ax
    )

    if title:
        fig.suptitle(title)
    plt.tight_layout(h_pad=0.0)
    return bp_ax, spg_ax

def plot_spectrogram_with_bp_rel2bl(spg_exp, spg_bl, band, bp_def, hyp, channel, start_time, end_time, title=None):
    b, s = _plot_spectrogram_with_bp_rel2bl(
        spg_exp.sel(channel=channel, time=slice(start_time, end_time)),
        spg_bl.sel(channel=channel, time=slice(start_time, end_time)),
        band,
        bp_def,
        hyp,
        title=title,
    )
    return b, s

def _plot_spectrogram_with_bandpower(spg, band_definition, band, hyp, title=None, figsize=(20, 5)):
    fig, (bp_ax, spg_ax) = plt.subplots(
        ncols=1,
        nrows=2,
        figsize=figsize,
        gridspec_kw=dict(width_ratios=[1], height_ratios=[1, 1]),
    )

    bp = kd.get_bp_set(spg, band_definition)[band]
    bp = kd.get_smoothed_da(bp, smoothing_sigma=14)
    sns.lineplot(x=bp.datetime, y=bp, color="black", ax=bp_ax)
    bp_ax.set(xlabel=None, ylabel='Raw '+band.capitalize()+' Power', xticks=[], xmargin=0)
    if hyp is not None:
        eplt.plot_hypnogram_overlay(
            hyp, xlim=bp_ax.get_xlim(), state_colors=state_colors, ax=bp_ax
        )

    eplt.plot_spectrogram(
        spg.frequency, spg.datetime, spg.values, yscale="log", f_range=(0, 50), ax=spg_ax
    )

    if title:
        fig.suptitle(title)
    plt.tight_layout(h_pad=0.0)
    return bp_ax, spg_ax


def plot_spectrogram_with_bandpower(spg, band_definition, band, hyp, channel, start_time, end_time, title=None, figsize=(20,5)):
    b, s = _plot_spectrogram_with_bandpower(
        spg.sel(channel=channel, datetime=slice(start_time, end_time)),
        band_definition,
        band,
        hyp,
        title=title,
        figsize=figsize
    )
    return b, s

def plot_spectrogram_kd(
    freqs,
    spg_times,
    spg,
    f_range=None,
    t_range=None,
    yscale="linear",
    figsize=(18, 6),
    vmin=None,
    vmax=None,
    title = 'Title',
    ax=None,
    ):
    freqs, spg_times, spg = dsps.trim_spectrogram(freqs, spg_times, spg, f_range, t_range)

    ax = dspu.check_ax(ax, figsize=figsize)
    im = ax.pcolormesh(spg_times, freqs, np.log10(spg), cmap='nipy_spectral', vmin=vmin, vmax=vmax, alpha=0.5, shading="gouraud")
    ax.figure.colorbar(im)
    ax.set_yscale(yscale)
    ax.set_ylabel("Frequency [Hz]")
    ax.set_xlabel("Time [sec]")
    ax.set_title(title)

    if yscale == "log":
        ax.set_ylim(np.min(freqs[freqs > 0]), np.max(freqs))
    return ax