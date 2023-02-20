from typing import Optional

import pandas as pd
import xarray as xr

from ..subjects import Subject
from ...projects import Project
from .... import utils as ece_utils

#####
# DataArray functions
#####


def gather_alias_dataarray(
    wneProject: Project,
    wneSubject: Subject,
    experiment: str,
    alias: str,
    probe: str,
    daExt: str,
) -> xr.DataArray:
    "Gather netCDF."
    assert daExt.endswith(".nc"), "Files to gather must use extension .nc"
    lfpTable = wneSubject.get_lfp_bin_table(experiment, alias, probe=probe)
    daFiles = wneProject.get_sglx_counterparts(
        wneSubject.name, lfpTable.path.values, daExt
    )
    daList = [xr.load_dataarray(f) for f in daFiles if f.is_file()]
    return xr.concat(daList, dim="time")


def remove_alias_dataarrays(
    wneProject: Project,
    wneSubject: Subject,
    experiment: str,
    alias: str,
    probe: str,
    daExt: str,
):
    "Remove netCDFs."
    assert daExt.endswith(".nc"), "Files to gather must use extension .nc"
    lfpTable = wneSubject.get_lfp_bin_table(experiment, alias, probe=probe)
    daFiles = wneProject.get_sglx_counterparts(
        wneSubject.name, lfpTable.path.values, daExt
    )
    for f in daFiles:
        f.unlink(missing_ok=True)


def gather_and_save_alias_dataarray(
    srcProject: Project,
    wneSubject: Subject,
    experiment: str,
    alias: str,
    probe: str,
    daExt: str,
    outputName: str,
    to_npy: bool = False,
    removeAfter: bool = False,
    destProject: Optional[Project] = None,
):
    """Gather netCDF, save as ONE NPY."""
    if destProject is None:
        destProject = srcProject
    da = gather_alias_dataarray(srcProject, wneSubject, experiment, alias, probe, daExt)
    saveDir = destProject.get_alias_subject_directory(
        experiment, alias, wneSubject.name
    )
    if to_npy:
        ece_utils.write_da_as_npy(da, outputName, saveDir)
    else:
        ece_utils.save_xarray(da, saveDir / outputName)
    if removeAfter:
        remove_alias_dataarrays(srcProject, wneSubject, experiment, alias, probe, daExt)


#####
# HTSV functions
#####


def gather_alias_htsv(
    wneProject: Project,
    wneSubject: Subject,
    experiment: str,
    alias: str,
    probe: str,
    ext: str,
) -> pd.DataFrame:
    lfpTable = wneSubject.get_lfp_bin_table(experiment, alias, probe=probe)
    htsvFiles = wneProject.get_sglx_counterparts(
        wneSubject.name, lfpTable.path.values, ext
    )
    dfs = [ece_utils.read_htsv(f) for f in htsvFiles if f.is_file()]
    return pd.concat(dfs).reset_index(drop=True)


def gather_and_save_alias_htsv(
    srcProject: Project,
    wneSubject: Subject,
    experiment: str,
    alias: str,
    probe: str,
    inExt: str,
    outFname: str,
    destProject: Optional[Project] = None,
):
    if destProject is None:
        destProject = srcProject
    df = gather_alias_htsv(srcProject, wneSubject, experiment, alias, probe, inExt)
    savefile = destProject.get_alias_subject_file(
        experiment, alias, wneSubject.name, outFname
    )
    ece_utils.write_htsv(df, savefile)
