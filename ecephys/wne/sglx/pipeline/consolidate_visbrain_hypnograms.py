import logging
import pandas as pd
import ecephys as ece
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def do_alias(wneProject, wneSubject, experiment, alias, probe):
    vbHgs = list()
    lfpTable = wneSubject.get_lfp_bin_table(experiment, alias, probe=probe)
    for lfpFile in tqdm(list(lfpTable.itertuples())):
        [vbFile] = wneProject.get_sglx_counterparts(
            wneSubject.name,
            [lfpFile.path],
            ece.wne.constants.VISBRAIN_EXT,
            remove_stream=True,
        )
        if vbFile.is_file():
            logger.debug(f"Loading file {vbFile.name}")
            vbHg = ece.hypnogram.FloatHypnogram.from_visbrain(vbFile)
            vbHg[["start_time", "end_time"]] += lfpFile.tExperiment
            vbHgs.append(vbHg)
        else:
            logger.warning(f"Hypnogram {vbFile.name} not found. Skipping.")

    if vbHgs:
        hg = ece.hypnogram.FloatHypnogram(pd.concat(vbHgs, ignore_index=True))
        hypnoFile = wneProject.get_alias_subject_file(
            experiment, alias, wneSubject.name, ece.wne.constants.HYPNOGRAM_FNAME
        )
        hg.write_htsv(hypnoFile)

        # TODO: Probably shouldn't do this, but kept for now for the sake of backwards compatibility.
        dtHypnoFile = wneProject.get_alias_subject_file(
            experiment,
            alias,
            wneSubject.name,
            ece.wne.constants.DATETIME_HYPNOGRAM_FNAME,
        )

        t0, dt0 = wneSubject.get_experiment_start(experiment, probe=probe)
        dtHg = hg.as_datetime(dt0)
        dtHg.write_htsv(dtHypnoFile)
