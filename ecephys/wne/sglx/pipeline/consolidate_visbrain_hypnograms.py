import logging
from typing import Optional

import pandas as pd
from tqdm.auto import tqdm

from ..subjects import Subject
from ...projects import Project
from ... import constants
from .... import hypnogram

logger = logging.getLogger(__name__)


def do_experiment_probe(
    wneProject: Project,
    wneSubject: Subject,
    experiment: str,
    probe: str,
):
    vbHgs = list()
    lfpTable = wneSubject.get_lfp_bin_table(experiment, probe=probe)
    for lfpFile in tqdm(list(lfpTable.itertuples())):
        [vbFile] = wneProject.get_sglx_counterparts(
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
