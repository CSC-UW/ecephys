import logging

import xarray as xr

from ecephys import utils
from ecephys.wne import Project
from ecephys.wne import constants

logger = logging.getLogger(__name__)


def open_lfps(
    project: Project,
    subject: str,
    experiment: str,
    probe: str,
    hotfix_times=True,
    drop_duplicate_times=False,
    xr_kwargs=dict(chunks="auto"),
):
    lf_file = project.get_experiment_subject_file(
        experiment, subject, f"{probe}{constants.LFP_EXT}"
    )
    lf = xr.open_dataarray(lf_file, engine="zarr", **xr_kwargs)
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
        utils.hotfix_times(lf.time.values)
    if drop_duplicate_times:
        lf = lf.drop_duplicates(dim="time", keep="first")
    return lf
