# TODO: Make and replace with gather_and_save_experiment_dataarray
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


def do_alias(
    opts: dict,
    destProject: SGLXProject,
    wneSubject: SGLXSubject,
    experiment: str,
    alias: Optional[str] = None,
    **kwargs
):
    lfpTable = wneSubject.get_lfp_bin_table(experiment, alias, **kwargs)

    for lfpFile in tqdm(list(lfpTable.itertuples())):
        [emgFile] = wne_sglx_utils.get_sglx_file_counterparts(
            destProject, wneSubject.name, [lfpFile.path], constants.EMG_EXT
        )
        sig = sglxr.load_trigger(
            lfpFile.path,
            opts["probes"][lfpFile.probe]["emgFromLfpChans"],
            t0=lfpFile.expmtPrbAcqFirstTime,  # TODO: Convert times before saving
            dt0=lfpFile.expmtPrbAcqFirstDatetime,
        )
        emg = xrsig.synthetic_emg(sig)
        utils.save_xarray(emg, emgFile)

    for probe in lfpTable.probe.unique():
        pipeline_utils.gather_and_save_alias_dataarray(
            destProject,
            wneSubject,
            experiment,
            alias,
            probe,
            constants.EMG_EXT,
            constants.EMG_FNAME,
        )
