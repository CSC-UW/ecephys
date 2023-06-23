import logging
import pathlib
from typing import Callable, Optional

import numpy as np
import pandas as pd

from ecephys import hypnogram
from ecephys import units
from ecephys import utils
from ecephys.sglx import file_mgmt
from ecephys.wne import constants
from ecephys.wne import Project
from ecephys.wne.sglx import sessions
from ecephys.wne.sglx import SGLXProject
from ecephys.wne.sglx import SGLXSubject

logger = logging.getLogger(__name__)


def float_hypnogram_to_datetime(
    subj: SGLXSubject, experiment: str, hyp: hypnogram.FloatHypnogram, hyp_prb: str
) -> hypnogram.DatetimeHypnogram:
    df = hyp._df.copy()
    df["start_time"] = subj.t2dt(experiment, hyp_prb, df["start_time"])
    df["end_time"] = subj.t2dt(experiment, hyp_prb, df["end_time"])
    df["duration"] = df["end_time"] - df["start_time"]
    return hypnogram.DatetimeHypnogram(df)


def datetime_hypnogram_to_float(
    subj: SGLXSubject, experiment: str, hyp: hypnogram.DatetimeHypnogram, hyp_prb: str
) -> hypnogram.FloatHypnogram:
    df = hyp._df.copy()
    df["start_time"] = subj.dt2t(experiment, hyp_prb, df["start_time"])
    df["end_time"] = subj.dt2t(experiment, hyp_prb, df["end_time"])
    df["duration"] = df["end_time"] - df["start_time"]
    return hypnogram.FloatHypnogram(df)


def get_sglx_file_counterparts(
    project: SGLXProject,
    subject: str,
    paths: list[pathlib.Path],
    extension: str,
    remove_probe: bool = False,
    remove_stream: bool = False,
) -> list[pathlib.Path]:
    """Get counterparts to SpikeGLX raw data files.

    Counterparts are mirrored at the project's subject directory, and likely
    have different suffixes than the original raw data files.

    Parameters:
    -----------
    project_name: str
        From projects.yaml
    subject_name: str
        Subject's name within this project, i.e. subject's directory name.
    paths: list of pathlib.Path
        The raw data files to get the counterparts of.
    extension:
        The extension to replace .bin or .meta with. See `replace_ftype`.

    Returns:
    --------
    list of pathlib.Path
    """
    counterparts = sessions.mirror_raw_data_paths(
        project.get_subject_directory(subject), paths
    )  # Mirror paths at the project's subject directory
    counterparts = [
        file_mgmt.replace_ftype(p, extension, remove_probe, remove_stream)
        for p in counterparts
    ]
    return utils.remove_duplicates(counterparts)


def load_datetime_hypnogram(
    project: Project,
    experiment: str,
    subject: SGLXSubject,
    simplify: bool = True,
) -> hypnogram.DatetimeHypnogram:
    hg = project.load_float_hypnogram(experiment, subject, simplify)
    params = project.load_experiment_subject_params(experiment, subject.name)
    return float_hypnogram_to_datetime(
        subject, experiment, hg, params["hypnogram_probe"]
    )


def load_singleprobe_sorting(
    sglxSortingProject: SGLXProject,
    sglxSubject: SGLXSubject,
    experiment: str,
    alias: str,
    probe: str,
    sorting: str = "sorting",
    postprocessing: str = "postpro",
    wneAnatomyProject: Optional[Project] = None,
    allow_no_sync_file=False,
) -> units.SpikeInterfaceKilosortSorting:
    if sorting is None:
        sorting = "sorting"
    if postprocessing is None:
        postprocessing = "postpro"

    # Get function for converting SI samples to imec0 timebase
    sample2time = sglxSortingProject.get_sample2time(
        sglxSubject.name,
        experiment,
        alias,
        probe,
        sorting,
        allow_no_sync_file=allow_no_sync_file,
    )

    # Load extractor
    extractor = sglxSortingProject.get_kilosort_extractor(
        sglxSubject.name,
        experiment,
        alias,
        probe,
        sorting,
        postprocessing=postprocessing,
    )

    # TODO: Why was this removed, and should it be restored?
    # extractor = units.si_ks_sorting.fix_isi_violations_ratio(extractor)

    # Add anatomy to the extractor, if available.
    if wneAnatomyProject is not None:
        anatomy_file = wneAnatomyProject.get_experiment_subject_file(
            experiment, sglxSubject.name, f"{probe}.structures.htsv"
        )
        assert anatomy_file.exists(), (
            f"Could not find anatomy file at: {anatomy_file}.\n"
            f"Set `wneAnatomyProject = None` in kwargs to ignore anatomy."
        )
        structs = utils.read_htsv(anatomy_file)
    else:
        # TODO: Passing np.Inf will break ephyviewer, when it attempts to plot all depths.
        structs = units.siutils.get_dummy_structure_table(lo=-np.Inf, hi=np.Inf)
    extractor = units.siutils.add_anatomy_properties_to_extractor(extractor, structs)

    return units.SpikeInterfaceKilosortSorting(extractor, sample2time)


def load_multiprobe_sorting(
    sglxSortingProject: SGLXProject,
    sglxSubject: SGLXSubject,
    experiment: str,
    alias: str,
    probes: list[str],
    sortings: dict[str, str] = None,
    postprocessings: dict[str, str] = None,
    wneAnatomyProject: Optional[Project] = None,
    allow_no_sync_file=False,
) -> units.MultiSIKS:
    if sortings is None:
        sortings = {prb: None for prb in probes}
    if postprocessings is None:
        postprocessings = {prb: None for prb in probes}

    return units.MultiSIKS(
        {
            probe: load_singleprobe_sorting(
                sglxSortingProject,
                sglxSubject,
                experiment,
                alias,
                probe=probe,
                sorting=sortings[probe],
                postprocessing=postprocessings[probe],
                wneAnatomyProject=wneAnatomyProject,
                allow_no_sync_file=allow_no_sync_file,
            )
            for probe in probes
        }
    )


def get_sample2time_lf(
    experiment_sync_table: pd.DataFrame, experiment_probe_ftable: pd.DataFrame
) -> Callable[[np.ndarray], np.ndarray]:
    assert len(experiment_probe_ftable["probe"].unique()) == 1, "Only one probe allowed"
    cum_samples_by_end = experiment_probe_ftable["nFileSamp"].cumsum()
    cum_samples_by_start = cum_samples_by_end.shift(1, fill_value=0)
    experiment_probe_ftable["start_sample"] = cum_samples_by_start
    experiment_probe_ftable["end_sample"] = cum_samples_by_end

    # Given a sample number in the original recording, we can now figure out:
    #   (1) the file it came from
    #   (3) how to map that file's times into our canonical timebase.
    # We make a function that does this for an arbitrary array of sample numbers, so we can use it later as needed.
    experiment_sync_table = experiment_sync_table.set_index("source")

    def sample2time(s):
        s = s.astype("float")
        t = np.empty(s.size, dtype="float")
        t[:] = np.nan  # Check a posteriori if we covered all input samples
        for file in experiment_probe_ftable.itertuples():
            mask = (s >= file.start_sample) & (
                s < file.end_sample
            )  # Mask samples belonging to this segment
            t[mask] = (
                s[mask] - file.start_sample
            ) / file.imSampRate + file.expmtPrbAcqFirstTime  # Convert to number of seconds in this probe's (expmtPrbAcq) timebase
            sync_entry = experiment_sync_table.loc[
                file.fname
            ]  # Get info needed to sync to imec0's (expmtPrbAcq) timebase
            t[mask] = (
                sync_entry.slope * t[mask] + sync_entry.intercept
            )  # Sync to imec0 (expmtPrbAcq) timebase
        assert not any(np.isnan(t)), (
            "Some of the provided sample indices were not covered by segments \n"
            "and therefore couldn't be converted to time"
        )

        return t

    return sample2time


def get_time2time_lf(
    experiment_sync_table: pd.DataFrame, experiment_probe_ftable: pd.DataFrame
) -> Callable[[np.ndarray], np.ndarray]:
    assert len(experiment_probe_ftable["probe"].unique()) == 1, "Only one probe allowed"
    experiment_sync_table = experiment_sync_table.set_index("source")

    def time2time(t1):
        t2 = np.full_like(t1, fill_value=np.nan)
        for file in experiment_probe_ftable.itertuples():
            mask = (t1 >= file.expmtPrbAcqFirstTime) & (
                t1 <= file.expmtPrbAcqLastTime
            )  # Mask samples belonging to this segment
            sync_entry = experiment_sync_table.loc[
                file.path.name
            ]  # Get info needed to sync to imec0's (expmtPrbAcq) timebase
            t2[mask] = (
                sync_entry.slope * t1[mask] + sync_entry.intercept
            )  # Sync to imec0 (expmtPrbAcq) timebase
        assert not any(np.isnan(t2)), (
            "Some of the provided sample indices were not covered by segments \n"
            "and therefore couldn't be converted to time"
        )

        return t2

    return time2time


def get_lf_time_synchronizer(
    sync_project: SGLXProject, sglx_subject: SGLXSubject, experiment: str, probe: str
) -> Callable[[np.ndarray], np.ndarray]:
    sync_file = sync_project.get_experiment_subject_file(
        experiment, sglx_subject.name, constants.LF_SYNC_FNAME
    )
    experiment_sync_table = utils.read_htsv(sync_file)
    experiment_probe_ftable = sglx_subject.get_experiment_frame(
        experiment, ftype="bin", stream="lf", probe=probe
    )
    return get_time2time_lf(experiment_sync_table, experiment_probe_ftable)
