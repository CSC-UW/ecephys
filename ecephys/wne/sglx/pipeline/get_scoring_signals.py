import json
import logging

import numpy as np
import pandas as pd
from pyedflib import highlevel as edf
from scipy import interpolate as interp
import xarray as xr

from ecephys import utils
from ecephys import wne

logger = logging.getLogger(__name__)


def resample(sig: xr.DataArray, target_fs: int) -> tuple(np.ndarray, np.ndarray):
    f = interp.interp1d(sig.time, sig.T, kind="cubic")
    new_times = np.arange(sig.time.min(), sig.time.max(), 1 / target_fs)
    new_data = f(new_times)
    return (new_times, new_data)


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


def prepare_lfp(lfp: xr.DataArray, target_fs: int) -> xr.DataArray:
    # We cannot resample using scipy.interp1d without removing duplicate timestamps
    lfp = lfp.drop_duplicates(dim="time", keep="first")  # Takes ~40s for 48h
    t, data = resample(lfp, target_fs)  # Takes ~5m for 48h
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    lbls = [f"LF{ch}" for ch in lfp.channel.values]
    return xr.DataArray(
        data,
        dims=("signal", "time"),
        coords={"signal": lbls, "time": t},
        attrs={"fs": target_fs, "units": lfp.units},
    )


def prepare_emg(emg: xr.DataArray, target_fs: int) -> xr.DataArray:
    emg = emg.drop_duplicates(dim="time", keep="first")
    t, data = resample(emg, target_fs)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    data = utils.clip_outliers(data)
    assert data.ndim == 1, "Expected exactly one dEMG"
    lbls = ["dEMG"]
    return xr.DataArray(
        np.atleast_2d(data),
        dims=("signal", "time"),
        coords={"signal": lbls, "time": t},
        attrs={"fs": target_fs, "units": emg.units},
    )


def write_edf_for_visbrain(
    lfp: xr.DataArray, emg: xr.DataArray, savefile, startdate=None
) -> bool:
    assert (
        lfp.fs == 100
    ), "LFP sampling rate must be 100Hz. If not, Visbrain will waste time resampling to 100Hz anyways."
    assert (
        emg.fs == 100
    ), "EMG sampling rate must be 100Hz. If not, Visbrain will waste time resampling to 100Hz anyways."
    lfp_t0 = float(lfp.time.min())
    emg_t0 = float(emg.time.min())
    assert np.allclose(
        lfp_t0, emg_t0, atol=0.100
    ), "LFP and EMG should start within 100ms of each other"
    lfp_hdrs = make_signal_headers_from_datarray(lfp, transducer="LFP")
    lfp_sigs = [lfp.sel(signal=hdr["label"]).values for hdr in lfp_hdrs]
    emg_hdrs = make_signal_headers_from_datarray(emg, transducer="dEMG")
    emg_sigs = [emg.sel(signal=hdr["label"]).values for hdr in emg_hdrs]
    signal_headers = lfp_hdrs + emg_hdrs
    signals = lfp_sigs + emg_sigs

    t0 = {
        "t0": {hdr["label"]: lfp_t0 for hdr in lfp_hdrs}
        | {hdr["label"]: emg_t0 for hdr in emg_hdrs}
    }
    file_header = edf.make_header(
        recording_additional=json.dumps(t0), startdate=startdate
    )
    return edf.write_edf(
        str(savefile), signals, signal_headers, file_header, digital=False
    )


def do_experiment(
    experiment: str,
    subj: wne.sglx.Subject,
    proj: wne.Project,
    probe: str,
    lfp_chans: list[int],
    write_bdf: bool = True,
):
    emg_file = proj.get_experiment_subject_file(experiment, subj.name, wne.EMG_FNAME)
    emg = xr.open_dataarray(emg_file)

    lfp_file = proj.get_experiment_subject_file(
        experiment, subj.name, f"{probe}.lf.zarr"
    )
    lfp = xr.open_dataarray(lfp_file)
    lfp = lfp.sel(channel=lfp_chans)

    target_fs = int(wne.VISBRAIN_FS)

    lfp_rs = prepare_lfp(lfp, target_fs)  # 20m for 48h
    lfp_rs.to_zarr(
        proj.get_experiment_subject_file(experiment, subj.name, wne.SCORING_LFP),
        mode="w",
    )

    emg_rs = prepare_emg(emg, target_fs)  # 1m for 48h
    emg_rs.to_zarr(
        proj.get_experiment_subject_file(experiment, subj.name, wne.SCORING_EMG),
        mode="w",
    )

    if write_bdf:
        bdf_file = proj.get_experiment_subject_file(
            experiment, subj.name, wne.SCORING_BDF
        )
        startdate = pd.Timestamp(lfp.datetime.min().values).to_pydatetime()
        write_edf_for_visbrain(lfp_rs, emg_rs, bdf_file, startdate=startdate)
