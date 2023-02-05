import logging
from tqdm.auto import tqdm
from typing import Optional
from . import utils
from ..subjects import Subject
from ...projects import Project
from ... import constants
from .... import utils as ece_utils
from .... import sglxr, xrsig

logger = logging.getLogger(__name__)

DEFAULT_EMG_OPTIONS = dict(
    target_sf=20,
    window_size=25.0,
    wp=[300, 600],
    ws=[275, 625],
    gpass=1,
    gstop=60,
    ftype="butter",
)


def do_alias(
    opts: dict,
    destProject: Project,
    wneSubject: Subject,
    experiment: str,
    alias: Optional[str] = None,
    **kwargs
):
    lfpTable = wneSubject.get_lfp_bin_table(experiment, alias, **kwargs)

    for lfpFile in tqdm(list(lfpTable.itertuples())):
        [emgFile] = destProject.get_sglx_counterparts(
            wneSubject.name, [lfpFile.path], constants.EMG_EXT
        )
        sig = sglxr.load_trigger(
            lfpFile.path,
            opts["probes"][lfpFile.probe]["emgFromLfpChans"],
            t0=lfpFile["expmtPrbAcqFirstTime"],  # TODO: Convert times before saving
            dt0=lfpFile["expmtPrbAcqFirstDatetime"],
        )
        emg = xrsig.LFPs(sig).synthetic_emg(**DEFAULT_EMG_OPTIONS)
        ece_utils.save_xarray(emg, emgFile)

    for probe in lfpTable.probe.unique():
        utils.gather_and_save_alias_dataarray(
            destProject,
            wneSubject,
            experiment,
            alias,
            probe,
            constants.EMG_EXT,
            constants.EMG_FNAME,
        )
