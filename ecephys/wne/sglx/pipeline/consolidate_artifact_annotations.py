import logging

import pandas as pd
from tqdm.auto import tqdm

from ecephys import utils
from ecephys.wne import constants
from ecephys.wne.sglx import SGLXProject
from ecephys.wne.sglx import SGLXSubject
from ecephys.wne.sglx import utils as wne_sglx_utils

logger = logging.getLogger(__name__)


def do_experiment_probe_stream(
    experiment: str,
    probe: str,
    stream: str,
    sglx_subject: SGLXSubject,
    data_project: SGLXProject,
    sync_project: SGLXProject,
):
    """
    Consolidate per-trigger-file artifacts.
    """

    raise NotImplementedError(
        """
        This needs discussion/resolution:
        - Using this function may overwrite previously existing manually annotated artifacts files
        - There's a conflict in definition of `start_time`/`end_time` between
        what this function does (start/end_time as experiment' canonical timebase) and what is currently the case in
        in existing artifact files, and what is expected by the sorting pipeline 
        (start/end_time as file-based).
        """
    )

    artifacts = list()
    ftab = sglx_subject.get_experiment_frame(
        experiment, stream=stream, ftype="bin", probe=probe
    )
    for bin_file in tqdm(list(ftab.itertuples())):
        [artifacts_file] = wne_sglx_utils.get_sglx_file_counterparts(
            data_project,
            sglx_subject.name,
            [bin_file.path],
            constants.ARTIFACTS_EXT,
        )
        logger.debug(f"Looking for file {artifacts_file.name}")
        if not artifacts_file.is_file():
            logger.debug("File not found.")
            continue
        df = pd.read_csv(artifacts_file)
        df["fname"] = bin_file.path.name
        # TODO: These two fields are currently left here purely for backwards compatibility, but should be removed, and downstream parts of the pipeline that use them should be updated load and use the artifact files directly if they need unsync'd times.
        df["expmtPrbAcqFirstTime"] = df["start_time"] + bin_file.expmtPrbAcqFirstTime
        df["expmtPrbAcqLastTime"] = df["end_time"] + bin_file.expmtPrbAcqFirstTime

        logger.debug(f"Converting file times to canonical timebase...")
        t2t = wne_sglx_utils.get_time_synchronizer(
            sync_project, sglx_subject, experiment, binfile=bin_file.path
        )
        df["start_time"] = t2t(df["start_time"] + bin_file.expmtPrbAcqFirstTime)
        df["end_time"] = t2t(df["end_time"] + bin_file.expmtPrbAcqFirstTime)
        df["duration"] = df["end_time"] - df["start_time"]

        artifacts.append(df)

    if artifacts:
        df = pd.concat(artifacts, ignore_index=True)
        outfile = data_project.get_experiment_subject_file(
            experiment,
            sglx_subject.name,
            f"{probe}.{stream}.{constants.ARTIFACTS_FNAME}",
        )
        utils.write_htsv(df, outfile)


def do_experiment(
    experiment: str,
    sglx_subject: SGLXSubject,
    data_project: SGLXProject,
    sync_project: SGLXProject,
):
    probes = sglx_subject.get_experiment_probes(experiment)
    for probe in probes:
        for stream in ["ap", "lf"]:
            do_experiment_probe_stream(
                experiment, probe, stream, sglx_subject, data_project, sync_project
            )
