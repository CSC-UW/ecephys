import logging
from typing import Optional

import pandas as pd
from tqdm.auto import tqdm

from ecephys import utils
from ecephys.wne import constants
from ecephys.wne.sglx import SGLXProject
from ecephys.wne.sglx import SGLXSubject
from ecephys.wne.sglx import utils as wne_sglx_utils

logger = logging.getLogger(__name__)


def do_probe_stream(
    srcProject: SGLXProject,
    destProject: SGLXProject,
    wneSubject: SGLXSubject,
    experiment: str,
    probe: str,
    stream: str,
):
    artifacts = list()
    ftab = wneSubject.get_experiment_frame(
        experiment, stream=stream, ftype="bin", probe=probe
    )
    for lfpfile in tqdm(list(ftab.itertuples())):
        [artfile] = wne_sglx_utils.get_sglx_file_counterparts(
            srcProject,
            wneSubject.name,
            [lfpfile.path],
            constants.ARTIFACTS_EXT,
        )
        logger.debug(f"Looking for file {artfile.name}")
        if not artfile.is_file():
            logger.debug("File not found.")
            continue
        df = pd.read_csv(artfile)
        df["fname"] = lfpfile.path.name
        # TODO: Sync to canonical timebase BEFORE saving
        df["expmtPrbAcqFirstTime"] = df["start_time"] + lfpfile.expmtPrbAcqFirstTime
        df["expmtPrbAcqLastTime"] = df["end_time"] + lfpfile.expmtPrbAcqFirstTime
        artifacts.append(df)

    if artifacts:
        df = pd.concat(artifacts, ignore_index=True)
        outfile = destProject.get_experiment_subject_file(
            experiment, wneSubject.name, f"{probe}.{stream}.{constants.ARTIFACTS_FNAME}"
        )
        utils.write_htsv(df, outfile)


def do_experiment(
    srcProject: SGLXProject,
    wneSubject: SGLXSubject,
    experiment: str,
    probes: Optional[list[str]] = None,
    destProject: Optional[SGLXProject] = None,
):
    if destProject is None:
        destProject = srcProject
    probes = wneSubject.get_experiment_probes(experiment) if probes is None else probes
    for probe in probes:
        for stream in ["ap", "lf"]:
            do_probe_stream(
                srcProject, destProject, wneSubject, experiment, probe, stream
            )
