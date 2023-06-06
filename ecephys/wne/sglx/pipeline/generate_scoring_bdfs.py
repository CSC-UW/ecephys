import pandas as pd
from tqdm.auto import tqdm
import xarray as xr

from ecephys.wne import constants
from ecephys.wne.sglx import SGLXProject
from ecephys.wne.sglx import SGLXSubject
from ecephys.wne.sglx import utils as wne_sglx_utils
from ecephys.wne.sglx.pipeline import get_scoring_signals


def do_experiment(
    experiment: str,
    wne_subject: SGLXSubject,
    project: SGLXProject,
    **kwargs,
):
    lfp_file = (
        project.get_experiment_subject_file(
            experiment, wne_subject.name, constants.SCORING_LFP
        ),
    )
    lfp = xr.open_dataarray(lfp_file, engine="zarr", chunks="auto")

    emg_file = (
        project.get_experiment_subject_file(
            experiment, wne_subject.name, constants.SCORING_EMG
        ),
    )
    emg = xr.open_dataarray(emg_file, engine="zarr", chunks="auto")

    lfp_table = wne_subject.get_lfp_bin_table(experiment, **kwargs)
    for lfp_file in tqdm(list(lfp_table.itertuples())):
        [visbrain_file] = wne_sglx_utils.get_sglx_file_counterparts(
            project,
            wne_subject.name,
            [lfp_file.path],
            constants.BDF_EXT,
            remove_stream=True,
        )
        visbrain_file.parent.mkdir(parents=True, exist_ok=True)
        file_slice = slice(lfp_file.expmtPrbAcqFirstTime, lfp_file.expmtPrbAcqLastTime)
        lfp_segment = lfp.sel(time=file_slice)
        emg_segment = emg.sel(time=file_slice)
        get_scoring_signals.write_edf_for_visbrain(
            lfp_segment,
            emg_segment,
            visbrain_file,
            startdate=pd.Timestamp(lfp_file.expmtPrbAcqFirstDatetime).to_pydatetime(),
        )
