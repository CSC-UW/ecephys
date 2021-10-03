from pathlib import Path
import hypnogram as hg
import pandas as pd
import yaml
import ecephys.xrsig as xrsig
import xarray as xr
import ecephys.plot as eplt
import matplotlib.pyplot as plt
import seaborn as sns

ANALYSIS_ROOT_DIRECTORY = Path(
    "N:\Data\paxilline_project_materials\PAX_6\PAX_6_analysis_data"
)

BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "beta": (13, 20),
    "gamma": (40, 80),
}


def assert_fname(fname):
    assert (
        isinstance(fname, str) or fname.parent == Path()
    ), "Need a filename, not a path."


def save_raw_data(sig, fname):
    #assert_fname(fname)
    path = ANALYSIS_ROOT_DIRECTORY / (fname + ".nc")
    sig.to_netcdf(path)
    sig.close()


def load_raw_data(fname):
    #assert_fname(fname)
    path = ANALYSIS_ROOT_DIRECTORY / (fname + ".nc")
    return xr.load_dataarray(path).swap_dims({"time": "datetime"})


def load_hypnograms(subject, experiment, condition, scoring_start_time):
    hypnograms_yaml_file = ANALYSIS_ROOT_DIRECTORY / "hypnogram_paths.yaml"

    with open(hypnograms_yaml_file) as fp:
        yaml_data = yaml.safe_load(fp)

    root = Path(yaml_data[subject]["hypno-root"])
    hypnogram_fnames = yaml_data[subject][experiment][condition]
    hypnogram_paths = [root / (fname + ".txt") for fname in hypnogram_fnames]

    hypnogram_start_times = pd.date_range(
        start=scoring_start_time, periods=len(hypnogram_paths), freq="7200S"
    )
    hypnograms = [
        hg.load_visbrain_hypnogram(path).as_datetime(start_time)
        for path, start_time in zip(hypnogram_paths, hypnogram_start_times)
    ]

    return pd.concat(hypnograms).reset_index(drop=True)


def save_hypnogram(hypnogram, fname):
    #assert_fname(fname)
    hypnogram_path = ANALYSIS_ROOT_DIRECTORY / (fname + ".hypnogram.tsv")
    hypnogram.write(hypnogram_path)


def load_hypnogram(fname):
    #assert_fname(fname)
    hypnogram_path = ANALYSIS_ROOT_DIRECTORY / (fname + ".hypnogram.tsv")
    return hg.load_datetime_hypnogram(hypnogram_path)


def get_spectrogram(sig):
    nperseg = int(4 * sig.fs)  # 4 second window
    noverlap = nperseg // 4  # 1 second overlap
    return xrsig.parallel_spectrogram_welch(sig, nperseg=nperseg, noverlap=noverlap)


def save_spectrogram(spg, fname):
    #assert_fname(fname)
    spg_path = ANALYSIS_ROOT_DIRECTORY / (fname + ".nc")
    spg.to_netcdf(spg_path)
    spg.close()


def load_spectrogram(fname):
    #assert_fname(fname)
    spg_path = ANALYSIS_ROOT_DIRECTORY / (fname + ".nc")
    return xr.load_dataarray(spg_path)


def _plot_spectrogram_with_bandpower(spg, bp, hyp, title=None, figsize=(22, 5)):
    state_colors = {
        "Wake": "palegreen",
        "Brief-Arousal": "palegreen",
        "Transition-to-NREM": "gainsboro",
        "Transition-to-Wake": "gainsboro",
        "Transition": "gainsboro",
        "NREM": "plum",
        "Transition-to-REM": "burlywood",
        "REM": "bisque",
        "Art": "crimson",
        "Unsure": "white",
    }

    fig, (bp_ax, spg_ax) = plt.subplots(
        ncols=1,
        nrows=2,
        figsize=figsize,
        gridspec_kw=dict(width_ratios=[1], height_ratios=[1, 1]),
    )

    # bp = get_smoothed_da(bp)
    sns.lineplot(x=bp.datetime.values, y=bp.values, color="black", ax=bp_ax)
    bp_ax.set(xlabel=None, ylabel="Bandpower", xticks=[], xmargin=0)
    eplt.plot_hypnogram_overlay(
        hyp, xlim=bp_ax.get_xlim(), state_colors=state_colors, ax=bp_ax
    )

    eplt.plot_spectrogram(
        spg.frequency, spg.datetime, spg, yscale="log", f_range=(0, 100), ax=spg_ax
    )

    if title:
        fig.suptitle(title)
    plt.tight_layout(h_pad=0.0)

    return (bp_ax, spg_ax)


def plot_spectrogram_with_bandpower(spg, bp, hyp, channel, start_time, end_time):
    start = hyp.start_time.min() + pd.to_timedelta(start_time)
    end = hyp.start_time.min() + pd.to_timedelta(end_time)
    return _plot_spectrogram_with_bandpower(
        spg.sel(channel=channel, datetime=slice(start, end)),
        bp.sel(channel=channel, datetime=slice(start, end)),
        hyp,
    )


def get_state_spectrogram(spg, hyp, state):
    return spg.sel(datetime=hyp.keep_states(state).covers_time(spg.datetime))


def get_state_psd(spg, hyp, state):
    return get_state_spectrogram(spg, hyp, state).median(dim="datetime")


def get_condition_psd(spg_name, hyp_name, state):
    hyp = load_hypnogram(hyp_name)
    #spg = load_spectrogram(fname)
    spg = load_spectrogram(spg_name).swap_dims({'time': 'datetime'})
    return get_state_psd(spg, hyp, state)


def compare_psd(
    psd1, psd2, keys=["condition1", "condition2"], key_name="condition", scale="log"
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
    g.set(xscale=scale, yscale=scale, ylabel='SWA as % of BL')
    return g