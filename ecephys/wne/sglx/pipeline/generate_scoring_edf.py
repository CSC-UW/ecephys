import ecephys as ece
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from scipy import interpolate as interp
from pyedflib import highlevel as edf


def resample(sig, target_fs):
    f = interp.interp1d(sig.time, sig.T, kind="cubic")
    new_times = np.arange(sig.time.min(), sig.time.max(), 1 / target_fs)
    new_data = f(new_times)
    return (new_times, new_data)


def normalize_1d(arr):
    arr = (arr - arr.min()) / (arr.max() - arr.min())  # Scales to between 0.0 and 1.0
    return (arr * 2.0) - 1.0  # Scales to between -1.0 and +1.0


def write_edf(sig, emg, edf_path):
    target_fs = 100.0
    sig_time_rs, sig_data_rs = resample(sig, target_fs)
    emg_time_rs, emg_data_rs = resample(emg, target_fs)

    signals = np.vstack([sig_data_rs, emg_data_rs])
    signals = np.nan_to_num(signals, nan=0.0, posinf=0.0, neginf=0.0)
    signals = np.apply_along_axis(normalize_1d, 1, signals)
    signal_labels = [f"LF{ch}" for ch in sig.channel.values] + ["dEMG"]

    signal_headers = edf.make_signal_headers(
        signal_labels,
        dimension=sig.units,
        sample_rate=target_fs,
        physical_min=signals.min(),
        physical_max=signals.max(),
    )
    edf.write_edf(str(edf_path), signals, signal_headers)


def do_file(lfpPath, emgPath, edfPath, scoringChans):
    lfp = ece.sglxr.load_trigger(lfpPath, scoringChans)
    emg = xr.load_dataarray(emgPath)
    write_edf(lfp, emg, edfPath)


# TODO: EDF has limited precision, and automatic determination of bit resolution often fails at long (e.g. 24h) timescales.
def do_alias(opts, destProject, wneSubject, experiment, alias, **kwargs):
    lfpTable = wneSubject.get_lfp_bin_table(experiment, alias, **kwargs)
    for lfpFile in tqdm(list(lfpTable.itertuples())):
        [emgFile] = destProject.get_sglx_counterparts(
            wneSubject.name, [lfpFile.path], ece.wne.constants.EMG_EXT
        )
        [edfFile] = destProject.get_sglx_counterparts(
            wneSubject.name,
            [lfpFile.path],
            ece.wne.constants.EDF_EXT,
            remove_stream=True,
        )
        edfFile.parent.mkdir(parents=True, exist_ok=True)
        chans = opts["probes"][lfpFile.probe]["scoringChans"]
        do_file(lfpFile.path, emgFile, edfFile, chans)


def check_lfp_resampling(sig, target_fs, time_rs, data_rs, n_seconds=4, iCh=0):
    fig, ax = plt.subplots(figsize=(36, 3))
    ax.plot(
        sig.time.values[: int(n_seconds * sig.fs)],
        sig.T.values[iCh, : int(n_seconds * sig.fs)],
    )
    ax.plot(
        time_rs[: int(n_seconds * target_fs)],
        data_rs[iCh, : int(n_seconds * target_fs)],
    )


def check_emg_resampling(emg, target_fs, time_rs, data_rs, n_seconds=4, iCh=0):
    fig, ax = plt.subplots(figsize=(36, 3))
    ax.plot(
        emg.time.values[: int(n_seconds * emg.target_sf)],
        emg.values[: int(n_seconds * emg.target_sf)],
    )
    ax.plot(
        time_rs[: int(n_seconds * target_fs)], data_rs[: int(n_seconds * target_fs)]
    )
