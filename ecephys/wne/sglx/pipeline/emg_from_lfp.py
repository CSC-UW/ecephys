# TODO: Save to zarr instead, and convert existing netCDF to zarr
import logging
from typing import Optional

from tqdm.auto import tqdm

from ecephys import sglxr
from ecephys import utils
from ecephys import xrsig
from ecephys.wne import constants
from ecephys.wne.sglx import SGLXProject
from ecephys.wne.sglx import SGLXSubject
from ecephys.wne.sglx import utils as wne_sglx_utils
from ecephys.wne.sglx.pipeline import utils as pipeline_utils


logger = logging.getLogger(__name__)


def do_experiment_probe(
    experiment: str,
    probe: str,
    sglx_subject: SGLXSubject,
    opts_project: SGLXProject,
    sync_project: SGLXProject,
    dest_project: SGLXProject,
    alias: Optional[str] = None,
):
    opts = opts_project.load_experiment_subject_params(experiment, sglx_subject.name)
    lfp_table = sglx_subject.get_lfp_bin_table(experiment, probe=probe, alias=alias)

    for lfp_file in tqdm(list(lfp_table.itertuples())):
        [emg_file] = wne_sglx_utils.get_sglx_file_counterparts(
            dest_project, sglx_subject.name, [lfp_file.path], constants.EMG_EXT
        )
        lfp = sglxr.load_trigger(
            lfp_file.path,
            opts["probes"][lfp_file.probe]["emgFromLfpChans"],
            t0=lfp_file.expmtPrbAcqFirstTime,
        )
        t2t = wne_sglx_utils.get_time_synchronizer(
            sync_project, sglx_subject, experiment, binfile=lfp_file.path
        )
        lfp = lfp.assign_coords({"time": t2t(lfp["time"].values)})
        emg = xrsig.synthetic_emg(lfp)
        utils.save_xarray_to_netcdf(emg, emg_file)

    pipeline_utils.gather_and_save_counterpart_netcdfs(
        dest_project,
        sglx_subject,
        experiment,
        probe,
        constants.EMG_EXT,
        constants.EMG_FNAME,
    )
