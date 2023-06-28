import numpy as np
from tqdm.auto import tqdm
import xarray as xr

from ecephys.wne import constants
from ecephys.wne.sglx import SGLXProject
from ecephys.wne.sglx import SGLXSubject
from ecephys.wne.sglx import utils as wne_sglx_utils
from ecephys.wne.sglx.pipeline import get_scoring_signals


def do_experiment_probe(
    experiment: str,
    probe: str,
    sglx_subject: SGLXSubject,
    data_project: SGLXProject,
    sync_project: SGLXProject,
):
    lfp_file = (
        data_project.get_experiment_subject_file(
            experiment, sglx_subject.name, constants.SCORING_LFP
        ),
    )
    lfp = xr.open_dataarray(lfp_file, engine="zarr", chunks="auto")

    emg_file = (
        data_project.get_experiment_subject_file(
            experiment, sglx_subject.name, constants.SCORING_EMG
        ),
    )
    emg = xr.open_dataarray(emg_file, engine="zarr", chunks="auto")

    lfp_table = sglx_subject.get_lfp_bin_table(experiment, probe=probe)
    for lfp_file in tqdm(list(lfp_table.itertuples())):
        [visbrain_file] = wne_sglx_utils.get_sglx_file_counterparts(
            data_project,
            sglx_subject.name,
            [lfp_file.path],
            constants.BDF_EXT,
            remove_stream=True,
        )
        visbrain_file.parent.mkdir(parents=True, exist_ok=True)
        t2t = wne_sglx_utils.get_time_synchronizer(
            sync_project, sglx_subject, experiment, binfile=lfp_file.path
        )
        (t1, t2) = t2t(
            np.asarray([lfp_file.expmtPrbAcqFirstTime, lfp_file.expmtPrbAcqLastTime])
        )
        lfp_segment = lfp.sel(time=slice(t1, t2))
        emg_segment = emg.sel(time=slice(t1, t2))
        get_scoring_signals.write_edf_for_visbrain(
            lfp_segment,
            emg_segment,
            visbrain_file,
        )
