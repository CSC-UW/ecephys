import logging

import pandas as pd
from tqdm.auto import tqdm

from ecephys import hypnogram
from ecephys.wne import constants
from ecephys.wne.sglx import SGLXProject
from ecephys.wne.sglx import SGLXSubject
from ecephys.wne.sglx import utils as wne_sglx_utils

logger = logging.getLogger(__name__)


def do_experiment_probe(
    wneProject: SGLXProject,
    wneSubject: SGLXSubject,
    experiment: str,
    probe: str,
):
    vbHgs = list()
    lfpTable = wneSubject.get_lfp_bin_table(experiment, probe=probe)
    for lfpFile in tqdm(list(lfpTable.itertuples())):
        [vbFile] = wne_sglx_utils.get_sglx_file_counterparts(
            wneProject,
            wneSubject.name,
            [lfpFile.path],
            constants.VISBRAIN_EXT,
            remove_stream=True,
        )
        if vbFile.is_file():
            logger.debug(f"Loading file {vbFile.name}")
            vbHg = hypnogram.FloatHypnogram.from_visbrain(vbFile)
            vbHg[
                ["start_time", "end_time"]
            ] += lfpFile.expmtPrbAcqFirstTime  # TODO: Sync times BEFORE saving
            vbHgs.append(vbHg)
        else:
            logger.warning(f"Hypnogram {vbFile.name} not found. Skipping.")

    if vbHgs:
        hg = pd.concat(vbHgs).sort_values("start_time").reset_index(drop=True)
        hg = hypnogram.FloatHypnogram.clean(hg)
        hypnoFile = wneProject.get_experiment_subject_file(
            experiment, wneSubject.name, constants.HYPNOGRAM_FNAME
        )
        hg.write_htsv(hypnoFile)
