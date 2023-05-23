# TODO: Save to zarr instead, and convert existing netCDF to zarr
# TODO: Convert times before saving
import logging

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


def do_experiment(
    opts: dict,
    dest_project: SGLXProject,
    wne_subject: SGLXSubject,
    experiment: str,
    **kwargs
):
    lfp_table = wne_subject.get_lfp_bin_table(experiment, **kwargs)

    for lfp_file in tqdm(list(lfp_table.itertuples())):
        [emg_file] = wne_sglx_utils.get_sglx_file_counterparts(
            dest_project, wne_subject.name, [lfp_file.path], constants.EMG_EXT
        )
        sig = sglxr.load_trigger(
            lfp_file.path,
            opts["probes"][lfp_file.probe]["emgFromLfpChans"],
            t0=lfp_file.expmtPrbAcqFirstTime,
            dt0=lfp_file.expmtPrbAcqFirstDatetime,
        )
        emg = xrsig.synthetic_emg(sig)
        utils.save_xarray(emg, emg_file)

    for probe in lfp_table.probe.unique():
        pipeline_utils.gather_and_save_counterpart_netcdfs(
            dest_project,
            wne_subject,
            experiment,
            probe,
            constants.EMG_EXT,
            constants.EMG_FNAME,
        )
