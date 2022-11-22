import ecephys as ece
import xarray as xr
import pandas as pd
from ..subjects import Subject
from ...projects import Project

#####
# DataArray functions
#####


def gather_alias_dataarray(wneProject, wneSubject, experiment, alias, probe, daExt):
    "Gather netCDF."
    assert daExt.endswith(".nc"), "Files to gather must use extension .nc"
    lfpTable = wneSubject.get_lfp_bin_table(experiment, alias, probe=probe)
    daFiles = wneProject.get_sglx_counterparts(
        wneSubject.name, lfpTable.path.values, daExt
    )
    daList = [xr.load_dataarray(f) for f in daFiles if f.is_file()]
    return xr.concat(daList, dim="time")


def remove_alias_dataarrays(wneProject, wneSubject, experiment, alias, probe, daExt):
    "Remove netCDFs."
    assert daExt.endswith(".nc"), "Files to gather must use extension .nc"
    lfpTable = wneSubject.get_lfp_bin_table(experiment, alias, probe=probe)
    daFiles = wneProject.get_sglx_counterparts(
        wneSubject.name, lfpTable.path.values, daExt
    )
    for f in daFiles:
        f.unlink(missing_ok=True)


def gather_and_save_alias_dataarray(
    wneProject,
    wneSubject,
    experiment,
    alias,
    probe,
    daExt,
    outputName,
    to_npy=False,
    removeAfter=False,
):
    """Gather netCDF, save as ONE NPY."""
    da = gather_alias_dataarray(wneProject, wneSubject, experiment, alias, probe, daExt)
    saveDir = wneProject.get_alias_subject_directory(experiment, alias, wneSubject.name)
    if to_npy:
        ece.utils.write_da_as_npy(da, outputName, saveDir)
    else:
        ece.utils.save_xarray(da, saveDir / outputName)
    if removeAfter:
        remove_alias_dataarrays(wneProject, wneSubject, experiment, alias, probe, daExt)


#####
# HTSV functions
#####


def gather_alias_htsv(
    wneProject: Project,
    wneSubject: Subject,
    experiment,
    alias,
    probe,
    ext,
):
    lfpTable = wneSubject.get_lfp_bin_table(experiment, alias, probe=probe)
    htsvFiles = wneProject.get_sglx_counterparts(
        wneSubject.name, lfpTable.path.values, ext
    )
    dfs = [ece.utils.read_htsv(f) for f in htsvFiles if f.is_file()]
    return pd.concat(dfs).reset_index(drop=True)


def gather_and_save_alias_htsv(
    wneProject: Project,
    wneSubject: Subject,
    experiment,
    alias,
    probe,
    inExt,
    outFname,
):
    df = gather_alias_htsv(wneProject, wneSubject, experiment, alias, probe, inExt)
    savefile = wneProject.get_alias_subject_file(
        experiment, alias, wneSubject.name, outFname
    )
    ece.utils.write_htsv(df, savefile)
