import json
import logging

import numpy as np
from pyedflib import highlevel as edf
import scipy.interpolate
import xarray as xr

from ecephys import utils
from ecephys.wne import constants
from ecephys.wne.sglx import SGLXProject
from ecephys.wne.sglx import SGLXSubject
import ecephys.wne.utils as wne_utils

logger = logging.getLogger(__name__)


def make_signal_headers_from_datarray(da: xr.DataArray, **kwargs) -> list:
    return [
        edf.make_signal_header(
            lbl,
            dimension=da.units,
            sample_rate=da.fs,
            physical_min=float(da.sel(signal=lbl).min()),
            physical_max=float(da.sel(signal=lbl).max()),
            **kwargs,
        )
        for lbl in da.signal.values
    ]


def prepare_lfp(lfp: xr.DataArray, new_times: np.ndarray[float], new_fs: float) -> xr.DataArray:
    """LFPs should have a label coord on the channel dim, which will be used for resulting signal headers"""
    # We cannot resample using scipy.interp1d without removing duplicate timestamps
    lfp = lfp.drop_duplicates(dim="time", keep="first")  # Takes ~40s for 48h

    # Resample data. Extrapolated values will be zero-filled.
    assert np.allclose(np.diff(new_times), 1 / new_fs), "New times do not match new sampling rate."
    f = scipy.interpolate.interp1d(lfp.time, lfp.T, kind="cubic", bounds_error=False, fill_value=0.0)
    data = f(new_times)  # Takes ~5m for 48h

    # Zero-fill any problematic values, remove outliers that would affect scaling
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    for i in range(data.shape[0]):
        data[i] = utils.clip_outliers(data[i], method="mad", n_devs=24)

    return xr.DataArray(
        data,
        dims=("signal", "time"),
        coords={"signal": lfp.label.values, "time": new_times},
        attrs={"fs": new_fs, "units": lfp.units},
    )


def prepare_emg(emg: xr.DataArray, new_times: np.ndarray[float], new_fs: float) -> xr.DataArray:
    # We cannot resample using scipy.interp1d without removing duplicate timestamps
    emg = emg.drop_duplicates(dim="time", keep="first")

    # Resample data. Extrapolated values will be zero-filled.
    assert np.allclose(np.diff(new_times), 1 / new_fs), "New times do not match new sampling rate."
    f = scipy.interpolate.interp1d(emg.time, emg.T, kind="cubic", bounds_error=False, fill_value=0.0)
    data = f(new_times)

    # Zero-fill any problematic values, remove outliers that would affect scaling
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    data = utils.clip_outliers(data, method="mad", n_devs=6)

    assert data.ndim == 1, "Expected exactly one dEMG"
    return xr.DataArray(
        np.atleast_2d(data),
        dims=("signal", "time"),
        coords={"signal": ["dEMG"], "time": new_times},
        attrs={"fs": new_fs, "units": emg.units},
    )


def write_edf_for_visbrain(lfp: xr.DataArray, emg: xr.DataArray, savefile, startdate=None) -> bool:
    assert (
        lfp.fs == 100
    ), "LFP sampling rate must be 100Hz. If not, Visbrain will waste time resampling to 100Hz anyways."
    assert (
        emg.fs == 100
    ), "EMG sampling rate must be 100Hz. If not, Visbrain will waste time resampling to 100Hz anyways."
    lfp_t0 = float(lfp.time.min())
    emg_t0 = float(emg.time.min())
    assert np.allclose(lfp_t0, emg_t0, atol=0.100), "LFP and EMG should start within 100ms of each other"
    ns = min(lfp.time.size, emg.time.size)  # The signals can still be off by a sample or two

    lfp_hdrs = make_signal_headers_from_datarray(lfp, transducer="LFP")
    lfp_sigs = [lfp.sel(signal=hdr["label"]).values[:ns] for hdr in lfp_hdrs]
    emg_hdrs = make_signal_headers_from_datarray(emg, transducer="dEMG")
    emg_sigs = [emg.sel(signal=hdr["label"]).values[:ns] for hdr in emg_hdrs]
    signal_headers = lfp_hdrs + emg_hdrs
    signals = lfp_sigs + emg_sigs

    t0 = {"t0": {hdr["label"]: lfp_t0 for hdr in lfp_hdrs} | {hdr["label"]: emg_t0 for hdr in emg_hdrs}}
    file_header = edf.make_header(recording_additional=json.dumps(t0), startdate=startdate)
    return edf.write_edf(str(savefile), signals, signal_headers, file_header, digital=False)


def do_experiment(
    experiment: str,
    sglx_subject: SGLXSubject,
    opts_project: SGLXProject,
    data_project: SGLXProject,
    write_bdf: bool = True,
):
    """
    An example `experiment_params.json` file:
    ...
    "scoring_signals": [
        {"probe": "imec4", "channel": 365, "label": "Sup. SSp"},
        {"probe": "imec4", "channel": 318, "label": "Deep SSp"},
        {"probe": "imec1", "channel": 173, "label": "HC"}
    ]
    ...
    """
    target_fs = int(constants.VISBRAIN_FS)

    # Load the scoring signals
    scoring_signals = opts_project.load_experiment_subject_params(experiment, sglx_subject.name)["scoring_signals"]
    lfps = [
        wne_utils.open_lfps(data_project, sglx_subject.name, experiment, d["probe"])
        .sel(channel=[d["channel"]])
        .assign_coords(label=("channel", [d["label"]]))
        for d in scoring_signals
    ]

    # Get the timestamps that signals will be resampled to.
    lf_min = min([lf.time.min() for lf in lfps])
    lf_max = max([lf.time.max() for lf in lfps])
    new_times = np.arange(lf_min, lf_max, 1 / target_fs)

    # Resample, tidy, and save.
    lfp_rs = xr.concat([prepare_lfp(lfp, new_times) for lfp in lfps], dim="signal")
    lfp_rs.to_zarr(
        data_project.get_experiment_subject_file(experiment, sglx_subject.name, constants.SCORING_LFP),
        mode="w",
    )

    # Do the same for EMGs
    emg_file = data_project.get_experiment_subject_file(experiment, sglx_subject.name, constants.EMG_FNAME)
    emg = xr.open_dataarray(emg_file)
    emg_rs = prepare_emg(emg, target_fs)  # 1m for 48h
    emg_rs.to_zarr(
        data_project.get_experiment_subject_file(experiment, sglx_subject.name, constants.SCORING_EMG),
        mode="w",
    )

    if write_bdf:
        bdf_file = data_project.get_experiment_subject_file(experiment, sglx_subject.name, constants.SCORING_BDF)
        write_edf_for_visbrain(lfp_rs, emg_rs, bdf_file)
