import logging
import warnings

import xarray as xr

import ecephys.utils
from ecephys import xrsig
from ecephys.wne.sglx import SGLXProject
from ecephys.wne import constants

logger = logging.getLogger(__name__)


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
    fname = (
        f"{fname_prefix}.{probe}{constants.LFP_EXT}"
        if fname_prefix is not None
        else f"{probe}{constants.LFP_EXT}"
    )
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
