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

    Because artifacts may have been specified at level of the file AND/OR
    at the level of the experiment (in a preexisting <probe>.<stream>.artifacts.htsv file
    in the experiment-subject directory), we aggregate entries from both sources.

    If for a given file, there are artifacts specified both in the trigger and in the 
    experiment file, we ensure that the trigger file contains all the artifacts.
    This ensures that artifacts specified directly in the experiment file will not
    be overriden.
    """

    artifacts = list()
    ftab = sglx_subject.get_experiment_frame(
        experiment, stream=stream, ftype="bin", probe=probe
    )

    # Preexisting experiment-lvel artifacts
    outfile = data_project.get_experiment_subject_file(
        experiment,
        sglx_subject.name,
        f"{probe}.{stream}.{constants.ARTIFACTS_FNAME}",
    )
    common_cols = ["withinFileStartTime", "withinFileEndTime", "type"]
    if outfile.exists():
        exp_artifacts = utils.read_htsv(outfile).loc[:,common_cols+["fname"]]
    else:
        exp_artifacts = pd.DataFrame([], columns=common_cols+["fname"])

    for bin_file in tqdm(list(ftab.itertuples())):

        fname = bin_file.path.name

        # Entries for this bin in preexisting consolidated artifact file
        exp_df = exp_artifacts.loc[
            exp_artifacts["fname"] == fname,
            common_cols
        ]

        # Entries for this bin in per-trigger artifact file
        [artifacts_file] = wne_sglx_utils.get_sglx_file_counterparts(
            data_project,
            sglx_subject.name,
            [bin_file.path],
            constants.ARTIFACTS_EXT,
        )
        if artifacts_file.is_file():
            trig_df = pd.read_csv(artifacts_file)
            assert set(trig_df.columns) == set(common_cols)

            # Avoid conflicts between old and new artifacts
            # If there's both trigger-level and preexisting experiment-level
            # artifacts for this bin, they should be identical
            comp_df = trig_df.merge(
                exp_df,indicator = True, how='outer'
            ).loc[lambda x : x['_merge']=='right_only']
            if len(comp_df):
                raise ValueError(
                    f"""Conflict between artifacts specified at the experiment-level in """
                    f""" \n`{outfile}`, \nand at the trigger-file level in \n{artifacts_file}. \n"""
                    f"""When a trigger artifact file exists, it should contain all the entries """
                    f"""for this file.  Please delete the following entries in the experiment file, """
                    f"""or copy them in the trigger file: \n{comp_df.loc[common_cols]}"""
                )

            # Merge old and new
            df = pd.concat([trig_df, exp_df]).drop_duplicates()

        else:
            df = exp_df

        if not len(df):
            continue

        df["fname"] = fname

        logger.debug(f"Converting file times to canonical timebase...")
        t2t = wne_sglx_utils.get_time_synchronizer(
            sync_project, sglx_subject, experiment, binfile=bin_file.path
        )
        df["start_time"] = t2t(df["withinFileStartTime"] + bin_file.expmtPrbAcqFirstTime)
        df["end_time"] = t2t(df["withinFileEndTime"] + bin_file.expmtPrbAcqFirstTime)
        df["duration"] = df["end_time"] - df["start_time"]

        artifacts.append(df)

    if artifacts:
        df = pd.concat(artifacts, ignore_index=True).sort_values(by="start_time")
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
