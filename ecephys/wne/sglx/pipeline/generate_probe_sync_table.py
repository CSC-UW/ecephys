import pathlib
import warnings

import numpy as np
import pandas as pd

from ecephys import sync
from ecephys import utils
from ecephys.wne import constants
from ecephys.wne.sglx import SGLXProject
from ecephys.wne.sglx import SGLXSubject
from ecephys.wne.sglx import utils as wne_sglx_utils


def get_barcode_sync_table(
    wneProject: SGLXProject,
    wneSubject: SGLXSubject,
    experiment: str,
    stream: str = "ap",
) -> pd.DataFrame:
    def load_barcodes(binpath: pathlib.Path) -> pd.DataFrame:
        [syncfile] = wne_sglx_utils.get_sglx_file_counterparts(
            wneProject, wneSubject.name, [binpath], constants.BARCODE_EXT
        )
        return utils.read_htsv(syncfile)

    probes = wneSubject.get_experiment_probes(experiment)
    ftabs = {
        probe: wneSubject.get_experiment_frame(
            experiment, stream=stream, ftype="bin", probe=probe
        )
        for probe in probes
    }
    for probe, ftab in ftabs.items():
        cols = ["session", "run", "gate", "trigger"]
        assert all(
            ftab[cols] == ftabs["imec0"][cols]
        ), "Files are not matched across probe tables"
        assert all(
            ftab.index == ftabs["imec0"].index
        ), "File indices are not matched across probe tables"
    nFiles = len(ftabs["imec0"])

    fits = pd.DataFrame()
    for i in range(nFiles):
        imec0BinPath = ftabs["imec0"].iloc[i]["path"]
        _df = pd.DataFrame(
            {
                "source": [imec0BinPath.name],
                "target": [imec0BinPath.name],
                "source_probe": "imec0",
                "target_probe": "imec0",
                "slope": [1.0],
                "intercept": [0.0],
            }
        )
        fits = pd.concat([fits, _df], ignore_index=True)
        for probe in set(probes) - {"imec0"}:
            thisBinPath = ftabs[probe].iloc[i]["path"]
            thisSyncData = load_barcodes(thisBinPath)
            imec0SyncData = load_barcodes(imec0BinPath)
            if min(len(thisSyncData), len(imec0SyncData)) == 0:
                warnings.warn(
                    f"Not enough barcodes to sync {thisBinPath} with {imec0BinPath}. Will attempt to interpolate."
                )
                _df = pd.DataFrame(
                    {
                        "source": [thisBinPath.name],
                        "target": [imec0BinPath.name],
                        "source_probe": probe,
                        "target_probe": "imec0",
                        "slope": [np.NaN],
                        "intercept": [np.NaN],
                    }
                )
                fits = pd.concat([fits, _df], ignore_index=True)
                continue

            fit = sync.fit_barcode_times(
                thisSyncData["time"].values,
                thisSyncData["value"].values,
                imec0SyncData["time"].values,
                imec0SyncData["value"].values,
                sysX_name=probe,
                sysY_name="imec0",
            )
            _df = pd.DataFrame(
                {
                    "source": [thisBinPath.name],
                    "target": [imec0BinPath.name],
                    "source_probe": probe,
                    "target_probe": "imec0",
                    "slope": [fit.coef_[0]],
                    "intercept": [fit.intercept_],
                }
            )
            fits = pd.concat([fits, _df], ignore_index=True)

    fits = fits.sort_values(["source_probe", "target_probe"])
    for probe in set(probes) - {"imec0"}:
        is_probe = fits["source_probe"] == probe
        fits.loc[is_probe] = fits.loc[is_probe].interpolate(
            method="linear", limit=1, limit_direction="both"
        )
    if fits["slope"].isna().any() or fits["intercept"].isna().any():
        raise ValueError(
            "Unable to interpolate missing sync info. Too much missing data for the given interpolation limit."
        )
    return fits


def do_experiment(
    wneProject: SGLXProject,
    wneSubject: SGLXSubject,
    experiment: str,
    stream: str = "ap",
):
    opts = wneProject.load_experiment_subject_json(
        experiment, wneSubject.name, constants.EXP_PARAMS_FNAME
    )
    if opts["imSyncType"] != "barcode":
        # TODO: Implement for single-probe / no-sync recordings [should be covered if implemented for square-pulse]
        raise NotImplementedError("Sync table generation only implemented for barcodes")

    sync_table = get_barcode_sync_table(wneProject, wneSubject, experiment, stream)

    f = wneProject.get_experiment_subject_file(
        experiment, wneSubject.name, f"prb_sync.{stream}.htsv"
    )
    utils.write_htsv(sync_table, f)
