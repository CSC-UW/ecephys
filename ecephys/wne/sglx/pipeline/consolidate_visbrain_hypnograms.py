import logging
import pandas as pd
from tqdm.auto import tqdm
from .. import subjects
from ... import projects
from .... import wne, hypnogram

logger = logging.getLogger(__name__)


def do_alias(
    srcProject: projects.Project,
    destProject: projects.Project,
    wneSubject: subjects.Subject,
    experiment: str,
    alias: str,
    probe: str,
):
    vbHgs = list()
    lfpTable = wneSubject.get_lfp_bin_table(experiment, alias, probe=probe)
    for lfpFile in tqdm(list(lfpTable.itertuples())):
        [vbFile] = srcProject.get_sglx_counterparts(
            wneSubject.name,
            [lfpFile.path],
            wne.constants.VISBRAIN_EXT,
            remove_stream=True,
        )
        if vbFile.is_file():
            logger.debug(f"Loading file {vbFile.name}")
            vbHg = hypnogram.FloatHypnogram.from_visbrain(vbFile)
            vbHg[["start_time", "end_time"]] += lfpFile.wneFileStartTime
            vbHgs.append(vbHg)
        else:
            logger.warning(f"Hypnogram {vbFile.name} not found. Skipping.")

    if vbHgs:
        hg = pd.concat(vbHgs).sort_values("start_time").reset_index(drop=True)
        hg = hypnogram.FloatHypnogram.clean(hg)
        hypnoFile = destProject.get_alias_subject_file(
            experiment, alias, wneSubject.name, wne.constants.HYPNOGRAM_FNAME
        )
        hg.write_htsv(hypnoFile)

        # Deprecated. Convert as needed using get_experiment_data_times.
        #
        # dtHypnoFile = destProject.get_alias_subject_file(
        #    experiment,
        #    alias,
        #    wneSubject.name,
        #    wne.constants.DATETIME_HYPNOGRAM_FNAME,
        # )

        # dt0, _ = wneSubject.get_experiment_data_times(
        #    experiment, probe, as_datetimes=True
        # )
        # dtHg = hg.as_datetime(dt0)
        # fdtHg.write_htsv(dtHypnoFile)
