import logging
from typing import Optional

import pandas as pd
from tqdm.auto import tqdm

from ecephys import hypnogram
from ecephys.wne.constants import FileExtensions, Files
from ecephys.wne.sglx import utils
from ecephys.wne.sglx.project import SGLXProject
from ecephys.wne.sglx.subject import SGLXSubject

logger = logging.getLogger(__name__)


def do_experiment_probe(
    experiment: str,
    probe: str,
    sglx_subject: SGLXSubject,
    sync_project: SGLXProject,
    data_project: SGLXProject,
    alias: str = "full",
) -> Optional[hypnogram.FloatHypnogram]:
    visbrain_hypnograms = list()
    lfp_table = sglx_subject.get_lfp_bin_table(experiment, probe=probe, alias=alias)
    for lfp_file in tqdm(list(lfp_table.itertuples())):
        [visbrain_hypnogram_file] = utils.get_sglx_file_counterparts(
            data_project,
            sglx_subject.name,
            [lfp_file.path],
            FileExtensions.VISBRAIN,
            remove_stream=True,
        )
        if visbrain_hypnogram_file.is_file():
            logger.debug(f"Loading file {visbrain_hypnogram_file.name}")
            visbrain_hypnogram = hypnogram.FloatHypnogram.from_visbrain(
                visbrain_hypnogram_file
            )
            logger.debug("Converting file times to canonical timebase...")
            t2t = utils.get_time_synchronizer(
                sync_project, sglx_subject, experiment, binfile=lfp_file.path
            )
            acq_t0 = utils.require_acq_time(lfp_file)
            visbrain_hypnogram["start_time"] = t2t(
                visbrain_hypnogram["start_time"] + acq_t0
            )
            visbrain_hypnogram["end_time"] = t2t(
                visbrain_hypnogram["end_time"] + acq_t0
            )
            visbrain_hypnogram["duration"] = (
                visbrain_hypnogram["end_time"] - visbrain_hypnogram["start_time"]
            )
            visbrain_hypnograms.append(visbrain_hypnogram)
        else:
            logger.warning(f"Hypnogram {visbrain_hypnogram_file} not found. Skipping.")

    if visbrain_hypnograms:
        hg = (
            pd.concat(visbrain_hypnograms)
            .sort_values("start_time")
            .reset_index(drop=True)
        )
        hg = hypnogram.FloatHypnogram.clean(hg)
        consolidated_hypnogram_file = data_project.get_experiment_subject_file(
            experiment, sglx_subject.name, f"{probe}.{Files.HYPNOGRAM}"
        )
        hg.write_htsv(consolidated_hypnogram_file)
        return hg
    else:
        return None
