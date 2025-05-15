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


def load_consolidated_hypnogram(
    project: Project,
    experiment: str,
    subject: str,
    probe: str,
    simplify: bool = True,
    clean: bool = True,
) -> hyp.FloatHypnogram:
    """Load FloatHypnogram from consolidated {probe}.hypnogram.htsv project file.
    All times are already in the canonical timebase.
    Cleaning has already been performed, but might need to be done again if simplying states.

    Important: This hypnogram might not be adequate for all use, as it does
    not necessarily account some excluded, missing or artifactual data
    from LF-band artifacts, AP-band artifacts, or sorting exclusions.  Consider
    using wisc_ecephys_tools.scoring.load_hypnogram instead.
    """
    f = project.get_experiment_subject_file(
        experiment, subject, f"{probe}.{Files.HYPNOGRAM}"
    )
    hg = hyp.FloatHypnogram.from_htsv(f)
    if simplify:
        hg = hg.replace_states(constants.SIMPLIFIED_STATES)
        if clean:
            hg = hyp.FloatHypnogram.clean(hg._df)
    return hg


def open_lfps(
    project: Project,
    subject: str,
    experiment: str,
    probe: str,
    hotfix_times=False,  # This costs almost nothing if there is no hotfixing to do.
    drop_duplicate_times=False,  # This is expensive no matter what. ~30s for 48h.
    chunks="auto",  # To force user-set chunks, use `chunks={}`
    anatomy_proj: Project = None,
    badchan_proj: Project = None,
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

    if badchan_proj is not None:
        params = badchan_proj.load_experiment_subject_params(experiment, subject)
        badchans = params["probes"][probe]["badChannels"]
        lf = lf.drop_sel({"channel": badchans})

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
