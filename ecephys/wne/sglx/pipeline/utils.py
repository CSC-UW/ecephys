from typing import Optional

import pandas as pd
import xarray as xr

from ecephys import utils
from ecephys.wne.sglx import SGLXProject
from ecephys.wne.sglx import SGLXSubject
from ecephys.wne.sglx import utils as wne_sglx_utils

#####
# DataArray functions
#####


def gather_counterpart_netcdfs(
    wne_project: SGLXProject,
    wne_subject: SGLXSubject,
    experiment: str,
    probe: str,
    da_ext: str,
) -> xr.DataArray:
    "Gather netCDF."
    assert da_ext.endswith(".nc"), "Files to gather must use extension .nc"
    lfp_table = wne_subject.get_lfp_bin_table(experiment, probe=probe)
    da_files = wne_sglx_utils.get_sglx_file_counterparts(
        wne_project, wne_subject.name, lfp_table.path.values, da_ext
    )
    da_list = [xr.load_dataarray(f) for f in da_files if f.is_file()]
    return xr.concat(da_list, dim="time")


def remove_counterpart_netcdfs(
    wne_project: SGLXProject,
    wne_subject: SGLXSubject,
    experiment: str,
    probe: str,
    da_ext: str,
):
    "Remove netCDFs."
    assert da_ext.endswith(".nc"), "Files to gather must use extension .nc"
    lfp_table = wne_subject.get_lfp_bin_table(experiment, probe=probe)
    da_files = wne_sglx_utils.get_sglx_file_counterparts(
        wne_project, wne_subject.name, lfp_table.path.values, da_ext
    )
    for f in da_files:
        f.unlink(missing_ok=True)


def gather_and_save_counterpart_netcdfs(
    src_project: SGLXProject,
    wne_subject: SGLXSubject,
    experiment: str,
    probe: str,
    da_ext: str,
    output_name: str,
    to_npy: bool = False,
    remove_after: bool = False,
    dest_project: Optional[SGLXProject] = None,
):
    """Gather netCDF, save as ONE NPY."""
    if dest_project is None:
        dest_project = src_project
    da = gather_counterpart_netcdfs(src_project, wne_subject, experiment, probe, da_ext)
    save_dir = dest_project.get_experiment_subject_directory(
        experiment, wne_subject.name
    )
    if to_npy:
        utils.write_da_as_npy(da, output_name, save_dir)
    else:
        utils.save_xarray_to_netcdf(da, save_dir / output_name)
    if remove_after:
        remove_counterpart_netcdfs(src_project, wne_subject, experiment, probe, da_ext)


#####
# HTSV functions
#####


def gather_alias_htsv(
    wneProject: SGLXProject,
    wneSubject: SGLXSubject,
    experiment: str,
    alias: str,
    probe: str,
    ext: str,
) -> pd.DataFrame:
    lfpTable = wneSubject.get_lfp_bin_table(experiment, alias, probe=probe)
    htsvFiles = wne_sglx_utils.get_sglx_file_counterparts(
        wneProject, wneSubject.name, lfpTable.path.values, ext
    )
    dfs = [utils.read_htsv(f) for f in htsvFiles if f.is_file()]
    return pd.concat(dfs).reset_index(drop=True)


def gather_and_save_alias_htsv(
    srcProject: SGLXProject,
    wneSubject: SGLXSubject,
    experiment: str,
    alias: str,
    probe: str,
    inExt: str,
    outFname: str,
    destProject: Optional[SGLXProject] = None,
):
    if destProject is None:
        destProject = srcProject
    df = gather_alias_htsv(srcProject, wneSubject, experiment, alias, probe, inExt)
    savefile = destProject.get_alias_subject_file(
        experiment, alias, wneSubject.name, outFname
    )
    utils.write_htsv(df, savefile)
