import logging
from ecephys import hypnogram
from ecephys.wne import Project
from ecephys.wne.sglx import SGLXProject
from ecephys.wne.sglx import SGLXSubject
import warnings

import xarray as xr
import pandas as pd

import ecephys.utils
from ecephys import xrsig
from ecephys.wne.sglx import SGLXProject
from ecephys.wne import constants

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


def load_consolidated_artifacts(
    project: Project, experiment: str, subject: str, probe: str, stream: str, simplify: bool = True
):
    artifacts_path = project.get_experiment_subject_file(
        experiment,
        subject,
        f"{probe}.{stream}.{constants.ARTIFACTS_FNAME}",
    )
    if artifacts_path.exists():
        artifacts = ecephys.utils.read_htsv(artifacts_path).loc[:, ["start_time", "end_time", "type"]]
    else:
        artifacts = pd.DataFrame([], columns=["start_time", "end_time", "type"])

    if simplify:
        return artifacts.replace(constants.SIMPLIFIED_ARTIFACTS)

    return artifacts


def load_raw_float_hypnogram(
    project: Project,
    experiment: str,
    subject: str,
    simplify: bool = True,
) -> hypnogram.FloatHypnogram:
    """Load FloatHypnogram from consolidated hypnogram.htsv project file.

    Important: This hypnogram might not be adequate for all use, as it does
    not necessarily account some excluded, missing or artifactual data
    from LF-band artifacts, AP-band artifacts, or sorting exclusions.  Consider
    using ecephys.wne.sglx.utils.load_reconciled_float_hypnogram instead.
    """
    f = project.get_experiment_subject_file(experiment, subject, constants.HYPNOGRAM_FNAME)
    hg = hypnogram.FloatHypnogram.from_htsv(f)
    if simplify:
        hg = hg.replace_states(constants.SIMPLIFIED_STATES)
    return hg


def load_raw_datetime_hypnogram(
    project: Project,
    experiment: str,
    subject: str,
    simplify: bool = True,
) -> hypnogram.DatetimeHypnogram:
    hg = load_raw_float_hypnogram(project, experiment, subject, simplify)
    params = project.load_experiment_subject_params(experiment, subject.name)
    return float_hypnogram_to_datetime(subject, experiment, hg, params["hypnogram_probe"])


def load_ephyviewer_hypnogram_edits(
    project: Project,
    experiment: str,
    subject: str,
    simplify: bool = True,
) -> pd.DataFrame:
    f = project.get_experiment_subject_file(experiment, subject, constants.HYPNOGRAM_EPHYVIEWER_EDITS_FNAME)
    df = pd.read_csv(f, sep=",")
    df = df.rename({"time": "start_time", "label": "state"}, axis=1)
    df["end_time"] = df["start_time"] + df["duration"]
    hg = hypnogram.FloatHypnogram(df)
    if simplify:
        hg = hg.replace_states(constants.SIMPLIFIED_STATES)
    return hg


def load_postprocessing_hypnogram_for_si_slicing(
    sglxSortingProject,
    sglxSubject: SGLXSubject,
    experiment: str,
    probe: str,
    alias: str = "full",
    sorting: str = "sorting",
    postprocessing: str = "postpro",
    drop_time_columns: bool = True,
) -> pd.DataFrame:
    """Load postprocessing hypnogram, which can be used with si.frame_slice

    Important:
    This is NOT adequate for use as regular hypnogram since the
    start/end_time and duration fields do not account for gaps!
    But the start_sample,end_sample columns can be used with
    the si.frame_slice() methods.
    However, this may be used as regular hypnogram after reconciliating with
    exclusions.
    """
    f = (
        sglxSortingProject.get_alias_subject_directory(experiment, alias, sglxSubject.name)
        / f"{sorting}.{probe}"
        / postprocessing
        / "hypnogram.htsv"
    )

    if not f.exists():
        import warnings

        warnings.warn(f"No `hypnogram.htsv` file in postpro dir. Returning None")
        return None

    df = ecephys.utils.read_htsv(f)
    if drop_time_columns:
        # Drop misleading start/end_time/duration columns
        return df.drop(columns=["start_time", "end_time", "duration"])

    return df


def open_lfps(
    project: SGLXProject,
    subject: str,
    experiment: str,
    probe: str,
    hotfix_times=False,
    drop_duplicate_times=False,
    chunks="auto",
    anatomy_proj: SGLXProject = None,
    fname_prefix: str = None,
    **xr_kwargs,
):
    fname = f"{fname_prefix}.{probe}{constants.LFP_EXT}" if fname_prefix is not None else f"{probe}{constants.LFP_EXT}"
    lf_file = project.get_experiment_subject_file(experiment, subject, fname)
    lf = xr.open_dataarray(lf_file, engine="zarr", chunks=chunks, **xr_kwargs)
    lf = lf.drop_vars("datetime", errors="ignore")
    # When loaded, attempting to access lf.chunksizes (or use fns that leverage chunking) will result in the following:
    # ValueError: Object has inconsistent chunks along dimension time. This can be fixed by calling unify_chunks().
    # This is because the datetime coordinate, despite being on the time dimension, has different chunksizes.
    # It is unclear how this happened. It seems that when the file was first created, lf.chunk({'time': 'auto}) was applied separately to each coord on the time dim.
    # As far as I can tell from inspecting lf.chunks, there are NOT inconsistent chunks along the time dimension of the data itself...
    # "Unifying" chunks will allow you to procede, but produces unequal chunksizes when it tries to reconcile the data time chunks with the datetime chunks. Instead, we can do this:
    if lf.chunks:
        try:
            lf.chunksizes
        except ValueError:
            # offending = find_unequal_chunks(lf, dim='time')
            logger.debug(
                "Xarray claims that chunk sizes are inconsistent. Rechunking using encoding['preferred chunks']..."
            )
            lf = lf.chunk(lf.encoding["preferred_chunks"])

    if hotfix_times:
        ecephys.utils.hotfix_times(lf.time.values)
    if drop_duplicate_times:
        lf = lf.drop_duplicates(dim="time", keep="first")

    # Add anatomy, if available
    if anatomy_proj is not None:
        anatomy_file = anatomy_proj.get_experiment_subject_file(experiment, subject, f"{probe}.structures.htsv")
        if anatomy_file.exists():
            structs = ecephys.utils.read_htsv(anatomy_file)
            lf = xrsig.assign_laminar_coordinate(lf, structs, sigdim="channel", lamdim="y")
        else:
            warnings.warn("Could not find anatomy file at: {anatomy_file}. Using dummy structure table")

    return lf
