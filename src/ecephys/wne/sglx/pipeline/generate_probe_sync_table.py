import pathlib
import warnings

import numpy as np
import pandas as pd

import ecephys.utils
from ecephys import sync
from ecephys.sglx.file_mgmt import resolve_reference_probe
from ecephys.wne.constants import FileExtensions
from ecephys.wne.sglx import utils
from ecephys.wne.sglx.project import SGLXProject
from ecephys.wne.sglx.subject import SGLXSubject


def do_experiment(
    experiment: str, sglx_subject: SGLXSubject, project: SGLXProject, stream: str = "ap"
):
    sessionIDs = sglx_subject.get_experiment_session_ids(experiment)
    for id in sessionIDs:
        do_session(id, sglx_subject, project, stream)

    experiment_sync_table = pd.concat(
        [
            ecephys.utils.read_htsv(
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
    ecephys.utils.write_htsv(experiment_sync_table, f)


def do_session(
    session_id: str, sglx_subject: SGLXSubject, project: SGLXProject, stream: str = "ap"
) -> pd.DataFrame:
    ftab = sglx_subject.get_session_frame(session_id, ftype="bin", stream=stream)
    sync_table = get_session_sync_table(project, sglx_subject, ftab)
    f = project.get_project_subject_file(
        sglx_subject.name, f"{session_id}.prb_sync.{stream}.htsv"
    )
    ecephys.utils.write_htsv(sync_table, f)


def get_reference_session_sync_table(ftab: pd.DataFrame, reference_probe: str):
    probes = ftab["probe"].unique()
    assert len(probes) == 1, "Expected only one probe"
    assert probes[0] == reference_probe, (
        f"Expected only the reference probe {reference_probe}, found {probes[0]}."
    )
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "source": [file.path.name],
                    "target": [file.path.name],
                    "source_probe": reference_probe,
                    "target_probe": reference_probe,
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
    probes, probe_ftabs, nFiles, reference_probe = _get_probe_ftabs(session_ftab)
    fits = get_reference_session_sync_table(
        probe_ftabs[reference_probe], reference_probe
    )
    for probe in set(probes) - {reference_probe}:
        for i in range(nFiles):
            probe_binpath = probe_ftabs[probe].iloc[i]["path"]
            print(f"Doing file {i}: {probe_binpath.name}")
            reference_binpath = probe_ftabs[reference_probe].iloc[i]["path"]

            if imSyncType == "barcode":
                slope, intercept = _get_barcode_file_fit(
                    project, sglx_subject, probe, probe_binpath, reference_binpath,
                    reference_probe,
                )
            elif imSyncType == "square_pulse":
                slope, intercept = _get_square_pulse_file_fit(
                    project, sglx_subject, probe, probe_binpath, reference_binpath,
                    reference_probe,
                )
            elif imSyncType == "random":
                slope, intercept = _get_random_pulse_file_fit(
                    project, sglx_subject, probe, probe_binpath, reference_binpath,
                    reference_probe,
                )
            file_fit = pd.DataFrame(
                {
                    "source": [probe_binpath.name],
                    "target": [reference_binpath.name],
                    "source_probe": probe,
                    "target_probe": reference_probe,
                    "slope": [slope],
                    "intercept": [intercept],
                }
            )
            fits = pd.concat([fits, file_fit], ignore_index=True)

    fits = fits.sort_values(["source_probe", "target_probe"])
    for probe in set(probes) - {reference_probe}:
        is_probe = fits["source_probe"] == probe
        is_na = fits[is_probe & fits[["slope", "intercept"]].isna().any(axis=1)]
        # Only interpolate if the file is so short that the lack of sync info to have capture sufficient sync pulses.
        if not is_na.empty:
            prb_ftab = probe_ftabs[probe].copy()
            prb_ftab["filename"] = prb_ftab.apply(lambda x: x.path.name, axis=1)
            prb_ftab = prb_ftab.set_index("filename")
            if all(prb_ftab.loc[is_na["source"], "fileTimeSecs"] < 60):
                fits.loc[is_probe] = fits.loc[is_probe].interpolate(method="linear")
    if fits["slope"].isna().any() or fits["intercept"].isna().any():
        raise ValueError("Unable to interpolate all the missing sync info.")
    return fits


def _load_ttls(
    project: SGLXProject, sglx_subject: SGLXSubject, binpath: pathlib.Path
) -> pd.DataFrame:
    [syncfile] = utils.get_sglx_file_counterparts(
        project, sglx_subject.name, [binpath], FileExtensions.TTL
    )
    return ecephys.utils.read_htsv(syncfile)


def _load_barcodes(
    project: SGLXProject, sglx_subject: SGLXSubject, binpath: pathlib.Path
) -> pd.DataFrame:
    [syncfile] = utils.get_sglx_file_counterparts(
        project, sglx_subject.name, [binpath], FileExtensions.BARCODE
    )
    return ecephys.utils.read_htsv(syncfile)


def _get_session_sync_type(session_ftab: pd.DataFrame) -> str:
    imSyncType = session_ftab["imSyncType"].values
    assert ecephys.utils.all_equal(imSyncType), (
        "Expected all session files to have the same sync type"
    )
    return imSyncType[0]


def _get_probe_ftabs(session_ftab: pd.DataFrame):
    probes = session_ftab["probe"].unique()
    reference_probe = resolve_reference_probe(probes)
    probe_ftabs = {
        probe: session_ftab[session_ftab["probe"] == probe].reset_index(drop=True)
        for probe in probes
    }

    for probe, tab in probe_ftabs.items():
        cols = ["session", "run", "gate", "trigger"]
        assert all(tab[cols] == probe_ftabs[reference_probe][cols]), (
            "Files are not matched across probe tables"
        )
        assert all(tab.index == probe_ftabs[reference_probe].index), (
            "File indices are not matched across probe tables"
        )

    nFiles = len(probe_ftabs[reference_probe])
    return probes, probe_ftabs, nFiles, reference_probe


def _get_barcode_file_fit(
    project: SGLXProject,
    sglx_subject: SGLXSubject,
    probe: str,
    probe_binpath: pathlib.Path,
    reference_binpath: pathlib.Path,
    reference_probe: str,
) -> tuple[float, float]:
    probe_barcodes = _load_barcodes(project, sglx_subject, probe_binpath)
    reference_barcodes = _load_barcodes(project, sglx_subject, reference_binpath)
    if min(len(probe_barcodes), len(reference_barcodes)) == 0:
        warnings.warn(
            f"Not enough barcodes to sync {probe_binpath.name} with {reference_binpath.name}. Will attempt to interpolate."
        )
        slope = np.nan
        intercept = np.nan
    else:
        fit = sync.fit_barcode_times(
            probe_barcodes["time"].values,
            probe_barcodes["value"].values,
            reference_barcodes["time"].values,
            reference_barcodes["value"].values,
            sysX_name=probe,
            sysY_name=reference_probe,
        )
        slope = fit.coef_[0]
        intercept = fit.intercept_
    return slope, intercept


def _get_square_pulse_file_fit(
    project: SGLXProject,
    sglx_subject: SGLXSubject,
    probe: str,
    probe_binpath: pathlib.Path,
    reference_binpath: pathlib.Path,
    reference_probe: str,
) -> tuple[float, float]:
    probe_ttls = _load_ttls(project, sglx_subject, probe_binpath)
    reference_ttls = _load_ttls(project, sglx_subject, reference_binpath)
    if min(len(probe_ttls), len(reference_ttls)) == 0:
        warnings.warn(
            f"Not enough TTLs to sync {probe_binpath.name} with {reference_binpath.name}. Will attempt to interpolate."
        )
        slope = np.nan
        intercept = np.nan
    else:
        fit = sync.fit_square_pulse_times(
            probe_ttls["rising"].values,
            probe_ttls["falling"].values,
            reference_ttls["rising"].values,
            reference_ttls["falling"].values,
            sysX_name=probe,
            sysY_name=reference_probe,
        )
        slope = fit.coef_[0]
        intercept = fit.intercept_
    return slope, intercept


def _get_random_pulse_file_fit(
    project: SGLXProject,
    sglx_subject: SGLXSubject,
    probe: str,
    probe_binpath: pathlib.Path,
    reference_binpath: pathlib.Path,
    reference_probe: str,
) -> tuple[float, float]:
    probe_ttls = _load_ttls(project, sglx_subject, probe_binpath)
    reference_ttls = _load_ttls(project, sglx_subject, reference_binpath)
    if min(len(probe_ttls), len(reference_ttls)) == 0:
        warnings.warn(
            f"Not enough TTLs to sync {probe_binpath.name} with {reference_binpath.name}. Will attempt to interpolate."
        )
        slope = np.nan
        intercept = np.nan
    else:
        fit = sync.fit_random_pulse_times(
            probe_ttls["rising"].values,
            reference_ttls["rising"].values,
            sysX_name=probe,
            sysY_name=reference_probe,
        )
        slope = fit.coef_[0]
        intercept = fit.intercept_
    return slope, intercept
