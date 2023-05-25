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
    src_project: SGLXProject,
    dest_project: SGLXProject,
    wne_subject: SGLXSubject,
    experiment: str,
    probe: str,
    stream: str,
):
    artifacts = list()
    ftab = wne_subject.get_experiment_frame(
        experiment, stream=stream, ftype="bin", probe=probe
    )
    for bin_file in tqdm(list(ftab.itertuples())):
        [artifacts_file] = wne_sglx_utils.get_sglx_file_counterparts(
            src_project,
            wne_subject.name,
            [bin_file.path],
            constants.ARTIFACTS_EXT,
        )
        logger.debug(f"Looking for file {artifacts_file.name}")
        if not artifacts_file.is_file():
            logger.debug("File not found.")
            continue
        df = pd.read_csv(artifacts_file)
        df["fname"] = bin_file.path.name
        # TODO: Sync to canonical timebase BEFORE saving
        df["expmtPrbAcqFirstTime"] = df["start_time"] + bin_file.expmtPrbAcqFirstTime
        df["expmtPrbAcqLastTime"] = df["end_time"] + bin_file.expmtPrbAcqFirstTime
        artifacts.append(df)

    if artifacts:
        df = pd.concat(artifacts, ignore_index=True)
        outfile = dest_project.get_experiment_subject_file(
            experiment,
            wne_subject.name,
            f"{probe}.{stream}.{constants.ARTIFACTS_FNAME}",
        )
        utils.write_htsv(df, outfile)


def do_experiment(
    src_project: SGLXProject,
    wne_subject: SGLXSubject,
    experiment: str,
    probes: Optional[list[str]] = None,
    dest_project: Optional[SGLXProject] = None,
):
    if dest_project is None:
        dest_project = src_project
    probes = wne_subject.get_experiment_probes(experiment) if probes is None else probes
    for probe in probes:
        for stream in ["ap", "lf"]:
            do_probe_stream(
                src_project, dest_project, wne_subject, experiment, probe, stream
            )
