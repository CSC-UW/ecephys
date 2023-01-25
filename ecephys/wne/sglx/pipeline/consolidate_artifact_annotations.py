import logging
import pandas as pd
import ecephys as ece
from tqdm.auto import tqdm
from ecephys import hypnogram as hg

logger = logging.getLogger(__name__)


def do_alias(srcProject, destProject, wneSubject, experiment, alias, probe):
    artifacts = list()
    lfpTable = wneSubject.get_lfp_bin_table(experiment, alias, probe=probe)
    for lfpFile in tqdm(list(lfpTable.itertuples())):
        [artFile] = srcProject.get_sglx_counterparts(
            wneSubject.name,
            [lfpFile.path],
            ece.wne.constants.ARTIFACTS_EXT,
            remove_stream=True,
        )
        logger.debug(f"Looking for file {artFile.name}")
        if not artFile.is_file():
            logger.debug("File not found.")
            continue
        df = pd.read_csv(artFile)
        df["duration"] = df.apply(lambda row: row.end_time - row.start_time, axis=1)
        df["state"] = "Artifact"
        df[["start_time", "end_time"]] += lfpFile.wneFileStartTime
        artifacts.append(df)

    if artifacts:
        df = hg.FloatHypnogram(pd.concat(artifacts, ignore_index=True))
        artFile = destProject.get_alias_subject_file(
            experiment, alias, wneSubject.name, ece.wne.constants.ARTIFACTS_FNAME
        )
        df.write_htsv(artFile)
