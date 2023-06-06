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
    wne_project: SGLXProject,
    wne_subject: SGLXSubject,
    experiment: str,
    probe: str,
):
    visbrain_hypnograms = list()
    lfp_table = wne_subject.get_lfp_bin_table(experiment, probe=probe)
    for lfp_file in tqdm(list(lfp_table.itertuples())):
        [visbrain_hypnogram_file] = wne_sglx_utils.get_sglx_file_counterparts(
            wne_project,
            wne_subject.name,
            [lfp_file.path],
            constants.VISBRAIN_EXT,
            remove_stream=True,
        )
        if visbrain_hypnogram_file.is_file():
            logger.debug(f"Loading file {visbrain_hypnogram_file.name}")
            visbrain_hypnogram = hypnogram.FloatHypnogram.from_visbrain(
                visbrain_hypnogram_file
            )
            visbrain_hypnogram[
                ["start_time", "end_time"]
            ] += lfp_file.expmtPrbAcqFirstTime  # TODO: Sync times BEFORE saving
            visbrain_hypnograms.append(visbrain_hypnogram)
        else:
            logger.warning(
                f"Hypnogram {visbrain_hypnogram_file.name} not found. Skipping."
            )

    if visbrain_hypnograms:
        hg = (
            pd.concat(visbrain_hypnograms)
            .sort_values("start_time")
            .reset_index(drop=True)
        )
        hg = hypnogram.FloatHypnogram.clean(hg)
        consolidated_hypnogram_file = wne_project.get_experiment_subject_file(
            experiment, wne_subject.name, constants.HYPNOGRAM_FNAME
        )
        hg.write_htsv(consolidated_hypnogram_file)
