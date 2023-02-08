import logging
import pandas as pd
from tqdm.auto import tqdm
from typing import Optional
from ..subjects import Subject
from ...projects import Project
from ... import constants
from .... import utils as ece_utils

logger = logging.getLogger(__name__)


def do_probe_stream(
    wneProject: Project, wneSubject: Subject, experiment: str, probe: str, stream: str
):
    artifacts = list()
    ftab = wneSubject.get_experiment_frame(
        experiment, stream=stream, ftype="bin", probe=probe
    )
    for lfpfile in tqdm(list(ftab.itertuples())):
        [artfile] = wneProject.get_sglx_counterparts(
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
        outfile = wneProject.get_experiment_subject_file(
            experiment, wneSubject.name, f"{probe}.{stream}.{constants.ARTIFACTS_FNAME}"
        )
        ece_utils.write_htsv(df, outfile)


def do_experiment(
    wneProject: Project,
    wneSubject: Subject,
    experiment: str,
    probes: Optional[list[str]] = None,
):
    probes = wneSubject.get_experiment_probes(experiment) if probes is None else probes
    for probe in probes:
        for stream in ["ap", "lf"]:
            do_probe_stream(wneProject, wneSubject, experiment, probe, stream)
