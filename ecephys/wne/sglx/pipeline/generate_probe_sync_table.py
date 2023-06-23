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


def do_experiment(
    project: SGLXProject, sglx_subject: SGLXSubject, experiment: str, stream: str = "ap"
):
    sessionIDs = sglx_subject.get_experiment_session_ids(experiment)
    for id in sessionIDs:
        do_session(project, sglx_subject, id, stream)

    experiment_sync_table = pd.concat(
        [
            utils.read_htsv(
                project.get_project_subject_file(
                    sglx_subject.name, f"{id}.prb_sync.{stream}.htsv"
                )
            )
            for id in sessionIDs
        ]
    )
    f = project.get_experiment_subject_file(
        experiment, sglx_subject.name, f"prb_sync.{stream}.htsv"
    )
    utils.write_htsv(experiment_sync_table, f)


def do_session(
    project: SGLXProject, sglx_subject: SGLXSubject, session_id: str, stream: str = "ap"
) -> pd.DataFrame:
    ftab = sglx_subject.get_session_frame(session_id, ftype="bin", stream=stream)
    sync_table = get_session_sync_table(project, sglx_subject, ftab)
    f = project.get_project_subject_file(
        sglx_subject.name, f"{session_id}.prb_sync.{stream}.htsv"
    )
    utils.write_htsv(sync_table, f)


def get_imec0_session_sync_table(ftab: pd.DataFrame):
    probes = ftab["probe"].unique()
    assert len(probes) == 1, "Expected only one probe"
    assert probes[0] == "imec0", "Only one probe found, and it is not imec0."
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "source": [file.path.name],
                    "target": [file.path.name],
                    "source_probe": "imec0",
                    "target_probe": "imec0",
                    "slope": [1.0],
                    "intercept": [0.0],
                }
            )
            for file in ftab.itertuples()
        ],
        ignore_index=True,
    )


def get_session_sync_table(
    project: SGLXProject, sglx_subject: SGLXSubject, session_ftab: pd.DataFrame
) -> pd.DataFrame:
    imSyncType = _get_session_sync_type(session_ftab)
    probes, probe_ftabs, nFiles = _get_probe_ftabs(session_ftab)
    fits = get_imec0_session_sync_table(probe_ftabs["imec0"])
    for probe in set(probes) - {"imec0"}:
        for i in range(nFiles):
            probe_binpath = probe_ftabs[probe].iloc[i]["path"]
            print(f"Doing file {i}: {probe_binpath.name}")
            imec0_binpath = probe_ftabs["imec0"].iloc[i]["path"]

            if imSyncType == "barcode":
                slope, intercept = _get_barcode_file_fit(
                    project, sglx_subject, probe, probe_binpath, imec0_binpath
                )
            elif imSyncType == "square_pulse":
                slope, intercept = _get_square_pulse_file_fit(
                    project, sglx_subject, probe, probe_binpath, imec0_binpath
                )
            elif imSyncType == "random":
                slope, intercept = _get_random_pulse_file_fit(
                    project, sglx_subject, probe, probe_binpath, imec0_binpath
                )
            file_fit = pd.DataFrame(
                {
                    "source": [probe_binpath.name],
                    "target": [imec0_binpath.name],
                    "source_probe": probe,
                    "target_probe": "imec0",
                    "slope": [slope],
                    "intercept": [intercept],
                }
            )
            fits = pd.concat([fits, file_fit], ignore_index=True)

    fits = fits.sort_values(["source_probe", "target_probe"])
    for probe in set(probes) - {"imec0"}:
        is_probe = fits["source_probe"] == probe
        if fits.loc[is_probe].isna().any().any():
            raise NotImplementedError(
                "Should only interpolate if file duration is too short for barcodes."
            )
            fits.loc[is_probe] = fits.loc[is_probe].interpolate(
                method="linear", limit=1, limit_direction="both"
            )
    if fits["slope"].isna().any() or fits["intercept"].isna().any():
        raise ValueError(
            "Unable to interpolate missing sync info. Too much missing data for the given interpolation limit."
        )
    return fits


def _load_ttls(
    project: SGLXProject, sglx_subject: SGLXSubject, binpath: pathlib.Path
) -> pd.DataFrame:
    [syncfile] = wne_sglx_utils.get_sglx_file_counterparts(
        project, sglx_subject.name, [binpath], constants.TTL_EXT
    )
    return utils.read_htsv(syncfile)


def _load_barcodes(
    project: SGLXProject, sglx_subject: SGLXSubject, binpath: pathlib.Path
) -> pd.DataFrame:
    [syncfile] = wne_sglx_utils.get_sglx_file_counterparts(
        project, sglx_subject.name, [binpath], constants.BARCODE_EXT
    )
    return utils.read_htsv(syncfile)


def _get_session_sync_type(session_ftab: pd.DataFrame) -> str:
    imSyncType = session_ftab["imSyncType"].values
    assert utils.all_equal(
        imSyncType
    ), "Expected all session files to have the same sync type"
    return imSyncType[0]


def _get_probe_ftabs(session_ftab: pd.DataFrame) -> dict[str, pd.DataFrame]:
    probes = session_ftab["probe"].unique()
    probe_ftabs = {
        probe: session_ftab[session_ftab["probe"] == probe].reset_index(drop=True)
        for probe in probes
    }

    for probe, tab in probe_ftabs.items():
        cols = ["session", "run", "gate", "trigger"]
        assert all(
            tab[cols] == probe_ftabs["imec0"][cols]
        ), "Files are not matched across probe tables"
        assert all(
            tab.index == probe_ftabs["imec0"].index
        ), "File indices are not matched across probe tables"

    nFiles = len(probe_ftabs["imec0"])
    return probes, probe_ftabs, nFiles


def _get_barcode_file_fit(
    project: SGLXProject,
    sglx_subject: SGLXSubject,
    probe: str,
    probe_binpath: pathlib.Path,
    imec0_binpath: pathlib.Path,
) -> tuple[float, float]:
    probe_barcodes = _load_barcodes(project, sglx_subject, probe_binpath)
    imec0_barcodes = _load_barcodes(project, sglx_subject, imec0_binpath)
    if min(len(probe_barcodes), len(imec0_barcodes)) == 0:
        warnings.warn(
            f"Not enough barcodes to sync {probe_binpath.name} with {imec0_binpath.name}. Will attempt to interpolate."
        )
        slope = np.NaN
        intercept = np.NaN
    else:
        fit = sync.fit_barcode_times(
            probe_barcodes["time"].values,
            probe_barcodes["value"].values,
            imec0_barcodes["time"].values,
            imec0_barcodes["value"].values,
            sysX_name=probe,
            sysY_name="imec0",
        )
        slope = fit.coef_[0]
        intercept = fit.intercept_
    return slope, intercept


def _get_square_pulse_file_fit(
    project: SGLXProject,
    sglx_subject: SGLXSubject,
    probe: str,
    probe_binpath: pathlib.Path,
    imec0_binpath: pathlib.Path,
) -> tuple[float, float]:
    probe_ttls = _load_ttls(project, sglx_subject, probe_binpath)
    imec0_ttls = _load_ttls(project, sglx_subject, imec0_binpath)
    if min(len(probe_ttls), len(imec0_ttls)) == 0:
        warnings.warn(
            f"Not enough TTLs to sync {probe_binpath.name} with {imec0_binpath.name}. Will attempt to interpolate."
        )
        slope = np.NaN
        intercept = np.NaN
    else:
        fit = sync.fit_square_pulse_times(
            probe_ttls["rising"].values,
            probe_ttls["falling"].values,
            imec0_ttls["rising"].values,
            imec0_ttls["falling"].values,
            sysX_name=probe,
            sysY_name="imec0",
        )
        slope = fit.coef_[0]
        intercept = fit.intercept_
    return slope, intercept


def _get_random_pulse_file_fit(
    project: SGLXProject,
    sglx_subject: SGLXSubject,
    probe: str,
    probe_binpath: pathlib.Path,
    imec0_binpath: pathlib.Path,
) -> tuple[float, float]:
    probe_ttls = _load_ttls(project, sglx_subject, probe_binpath)
    imec0_ttls = _load_ttls(project, sglx_subject, imec0_binpath)
    if min(len(probe_ttls), len(imec0_ttls)) == 0:
        warnings.warn(
            f"Not enough TTLs to sync {probe_binpath.name} with {imec0_binpath.name}. Will attempt to interpolate."
        )
        slope = np.NaN
        intercept = np.NaN
    else:
        fit = sync.fit_random_pulse_times(
            probe_ttls["rising"].values,
            imec0_ttls["rising"].values,
            sysX_name=probe,
            sysY_name="imec0",
        )
        slope = fit.coef_[0]
        intercept = fit.intercept_
    return slope, intercept
