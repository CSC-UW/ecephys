import logging
import warnings

import pandas as pd
import xarray as xr

import ecephys.utils
from ecephys import hypnogram as hyp
from ecephys import xrsig

from . import constants
from .constants import FileExtensions as Exts
from .constants import Files
from .project import Project
from .subject import Subject

logger = logging.getLogger(__name__)


def float_hypnogram_to_datetime(
    subj: Subject, experiment: str, hyp: hyp.FloatHypnogram, hyp_prb: str
) -> hyp.DatetimeHypnogram:
    df = hyp._df.copy()
    df["start_time"] = subj.t2dt(experiment, hyp_prb, df["start_time"])
    df["end_time"] = subj.t2dt(experiment, hyp_prb, df["end_time"])
    df["duration"] = df["end_time"] - df["start_time"]
    return hyp.DatetimeHypnogram(df)


def datetime_hypnogram_to_float(
    subj: Subject, experiment: str, hyp: hyp.DatetimeHypnogram, hyp_prb: str
) -> hyp.FloatHypnogram:
    df = hyp._df.copy()
    df["start_time"] = subj.dt2t(experiment, hyp_prb, df["start_time"])
    df["end_time"] = subj.dt2t(experiment, hyp_prb, df["end_time"])
    df["duration"] = df["end_time"] - df["start_time"]
    return hyp.FloatHypnogram(df)


# TODO: This seems SLGX-specific, since it uses f"{probe}.{stream}.{Files.ARTIFACTS}".
# And, it is only used in sglx.utils.load_sglx_inclusions_and_artifacts(). Move it there?
def load_consolidated_artifacts(
    project: Project,
    experiment: str,
    subject: str,
    probe: str,
    stream: str,
    simplify: bool = True,
):
    artifacts_path = project.get_experiment_subject_file(
        experiment,
        subject,
        f"{probe}.{stream}.{Files.ARTIFACTS}",
    )
    if artifacts_path.exists():
        artifacts = ecephys.utils.read_htsv(artifacts_path).loc[
            :, ["start_time", "end_time", "type"]
        ]
    else:
        artifacts = pd.DataFrame([], columns=["start_time", "end_time", "type"])

    if simplify:
        return artifacts.replace(constants.SIMPLIFIED_ARTIFACTS)

    return artifacts


# TODO: This whole function appears to be an unnecessary duplication of ecephys.wne.projects.Project.load_float_hypnogram()
# Of the two, this is the one that is used, but which should be preferred? Probably this one.
def load_raw_float_hypnogram(
    project: Project,
    experiment: str,
    subject: str,
    simplify: bool = True,
) -> hyp.FloatHypnogram:
    """Load FloatHypnogram from consolidated hypnogram.htsv project file.

    Important: This hypnogram might not be adequate for all use, as it does
    not necessarily account some excluded, missing or artifactual data
    from LF-band artifacts, AP-band artifacts, or sorting exclusions.  Consider
    using ecephys.wne.sglx.utils.load_reconciled_float_hypnogram instead.
    """
    f = project.get_experiment_subject_file(experiment, subject, Files.HYPNOGRAM)
    hg = hyp.FloatHypnogram.from_htsv(f)
    if simplify:
        hg = hg.replace_states(constants.SIMPLIFIED_STATES)
        # TODO: This clean() should not be necessary. It is already done in ecephys.wne.sglx.pipeline.consoldate_visbrain_hypnograms.do_experiment_probe().
        # Although, it will change NaNs to NoData. Is that expected downstream somewhere?
        hg = hyp.FloatHypnogram.clean(hg._df)
    return hg


# TODO: This seems like it belongs in wisc_ecephys_tools
def load_ephyviewer_hypnogram_edits(
    project: Project,
    experiment: str,
    subject: str,
    simplify: bool = True,
) -> pd.DataFrame:
    f = project.get_experiment_subject_file(
        experiment, subject, Files.HYPNOGRAM_EPHYVIEWER_EDITS
    )
    if not f.exists():
        return hyp.FloatHypnogram(
            pd.DataFrame([], columns=["state", "start_time", "end_time", "duration"])
        )

    df = pd.read_csv(f, sep=",")
    df = df.rename({"time": "start_time", "label": "state"}, axis=1)
    df["end_time"] = df["start_time"] + df["duration"]
    hg = hyp.FloatHypnogram(df)
    if simplify:
        hg = hg.replace_states(constants.SIMPLIFIED_STATES)
    return hyp.FloatHypnogram(hyp.condense(hg._df, 0.1))


def open_lfps(
    project: Project,
    subject: str,
    experiment: str,
    probe: str,
    hotfix_times=False,  # This costs almost nothing if there is no hotfixing to do.
    drop_duplicate_times=False,  # This is expensive no matter what. ~30s for 48h.
    chunks="auto",
    anatomy_proj: Project = None,
    fname_prefix: str = None,
    **xr_kwargs,
):
    fname = (
        f"{fname_prefix}.{probe}{Exts.LFP}"
        if fname_prefix is not None
        else f"{probe}{Exts.LFP}"
    )
    lf_file = project.get_experiment_subject_file(experiment, subject, fname)
    lf = xr.open_dataarray(lf_file, engine="zarr", chunks=chunks, **xr_kwargs)
    lf = lf.drop_vars("datetime", errors="ignore")
    # When loading old files, attempting to access lf.chunksizes
    # (or use fns that leverage chunking) may result in the following:
    # ValueError: Object has inconsistent chunks along dimension time.
    # This is because when the file was first created, lf.chunk({'time': 'auto})
    # was used, and chunk sizes were determined separately for each coord on the
    # time dimension. There are not inconsistent chunks along the time dimension
    # of the data itself.
    # See: https://github.com/pydata/xarray/discussions/8037
    # This has since been fixed upstream, so new files will not have this issue.
    # Calling unify_chunks() will allow you to procede, but produces unequal
    # chunksizes when it tries to reconcile the data time chunks with the datetime
    # chunks. Instead, we can do this:
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
        anatomy_file = anatomy_proj.get_experiment_subject_file(
            experiment, subject, f"{probe}.structures.htsv"
        )
        if anatomy_file.exists():
            structs = ecephys.utils.read_htsv(anatomy_file)
            lf = xrsig.assign_laminar_coordinate(
                lf, structs, sigdim="channel", lamdim="y"
            )
        else:
            warnings.warn(
                "Could not find anatomy file at: {anatomy_file}. Using dummy structure table"
            )

    return lf


def get_dummy_artifacts_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "withinFileStartTime": [],
            "withinFileEndTime": [],
            "type": [],
            "fname": [],
            "start_time": [],
            "end_time": [],
            "duration": [],
        }
    )
