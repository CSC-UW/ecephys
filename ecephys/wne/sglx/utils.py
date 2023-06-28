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
    if wneAnatomyProject is None:
        wneAnatomyProject = sglxSortingProject

    anatomy_file = wneAnatomyProject.get_experiment_subject_file(
        experiment, sglxSubject.name, f"{probe}.structures.htsv"
    )
    if anatomy_file.exists():
        structs = utils.read_htsv(anatomy_file)
    else:
        import warnings
        warnings.warn("Could not find anatomy file at: {anatomy_file}. Using dummy structure table")
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


def get_experiment_sample2time(
    experiment_sync_table: pd.DataFrame, experiment_probe_ftable: pd.DataFrame
) -> Callable[[np.ndarray], np.ndarray]:
    """Get a function that maps samples in the original recording to the canonical timebase.
    WARNING: This is not appropriate for use with recordings where pieces have been excised and the bookkeeping has not been done to keep track of the excisions.
    """
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


def get_time2time(
    experiment_sync_table: pd.DataFrame,
    experiment_probe_ftable: pd.DataFrame,
    binfile: Optional[pathlib.Path] = None,
    extrapolate: bool = False,
) -> Callable[[np.ndarray], np.ndarray]:
    assert (
        len(experiment_probe_ftable["probe"].unique()) == 1
    ), "Cannot generate a time2time function without knowing the probe"
    experiment_sync_table = experiment_sync_table.set_index("source")

    if binfile is not None:
        # If we know the file a-priori, we can give a maximally precise time2time function
        def file_time2time(t1):
            sync_entry = experiment_sync_table.loc[binfile.name]
            return sync_entry.slope * t1 + sync_entry.intercept

        return file_time2time

    else:
        # If we don't know the binfile a-priori, our time-to-time function has to infer it.
        # WARNING: Because of file overlap, this method of assigning times to files is imperfect! Use per-file sync for maximum precision!
        def experiment_time2time(t1):
            t2 = np.full_like(t1, fill_value=np.nan)
            for file in experiment_probe_ftable.itertuples():
                mask = (t1 >= file.expmtPrbAcqFirstTime) & (
                    t1 <= file.expmtPrbAcqLastTime + (1 / file.imSampRate)
                )  # Mask samples belonging to this file
                sync_entry = experiment_sync_table.loc[
                    file.path.name
                ]  # Get info needed to sync to imec0's (expmtPrbAcq) timebase
                t2[mask] = (
                    sync_entry.slope * t1[mask] + sync_entry.intercept
                )  # Sync to imec0 (expmtPrbAcq) timebase
            is_nan = np.isnan(t2)
            if any(is_nan):
                msg = "Some of the provided times were not covered by the original recording and therefore can't be converted unambiguously."
                if extrapolate:
                    logger.warning(msg + " Using sync info from the nearest file.")
                    allowed_times = experiment_probe_ftable[
                        ["expmtPrbAcqFirstTime", "expmtPrbAcqLastTime"]
                    ].values.flatten()
                    allowed_files = experiment_probe_ftable[
                        ["path", "path"]
                    ].values.flatten()
                    for ix_t in np.where(is_nan)[0]:
                        nearest_allowed = utils.find_nearest(allowed_times, t1[ix_t])
                        nearest_fname = allowed_files[nearest_allowed].name
                        sync_entry = experiment_sync_table.loc[nearest_fname]
                        t2[ix_t] = sync_entry.slope * t1[ix_t] + sync_entry.intercept
                else:
                    raise ValueError(msg)

            return t2

        return experiment_time2time


def get_time_synchronizer(
    sync_project: SGLXProject,
    sglx_subject: SGLXSubject,
    experiment: str,
    stream: Optional[str] = None,
    probe: Optional[str] = None,
    binfile: Optional[pathlib.Path] = None,
    extrapolate: bool = False,
) -> Callable[[np.ndarray], np.ndarray]:
    if binfile is not None:
        (_, _, _, probe_, stream_, _) = file_mgmt.parse_sglx_fname(binfile.name)
        probe = probe_ if probe is None else probe
        assert probe == probe_, "Mismatch between provided probe and binfile"
        stream = stream_ if stream is None else stream
        assert stream == stream_, "Mismatch between provided stream and binfile"
    assert probe is not None, "Must provide probe"
    assert stream is not None, "Must provide stream"
    experiment_probe_ftable = sglx_subject.get_experiment_frame(
        experiment, ftype="bin", stream=stream, probe=probe
    )
    experiment_sync_table = utils.read_htsv(
        sync_project.get_experiment_subject_file(
            experiment, sglx_subject.name, constants.SYNC_FNAME_MAP[stream]
        )
    )
    return get_time2time(
        experiment_sync_table, experiment_probe_ftable, binfile, extrapolate
    )
