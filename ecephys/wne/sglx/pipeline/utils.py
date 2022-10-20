import ecephys as ece
import xarray as xr


def gather_alias_dataarray(wneProject, wneSubject, experiment, alias, probe, daExt):
    "Gather netCDF."
    assert daExt.endswith(".nc"), "Files to gather must use extension .nc"
    lfpTable = wneSubject.get_lfp_bin_table(experiment, alias, probe=probe)
    daFiles = wneProject.get_sglx_counterparts(
        wneSubject.name, lfpTable.path.values, daExt
    )
    daList = [xr.load_dataarray(f) for f in daFiles if f.is_file()]
    return xr.concat(daList, dim="time")


def gather_and_save_alias_dataarray(
    wneProject, wneSubject, experiment, alias, probe, daExt, outputName, to_npy=False
):
    """Gather netCDF, save as ONE NPY."""
    da = gather_alias_dataarray(wneProject, wneSubject, experiment, alias, probe, daExt)
    saveDir = wneProject.get_alias_subject_directory(experiment, alias, wneSubject.name)
    if to_npy:
        ece.utils.write_da_as_npy(da, outputName, saveDir)
    else:
        ece.utils.save_xarray(da, saveDir / outputName)
