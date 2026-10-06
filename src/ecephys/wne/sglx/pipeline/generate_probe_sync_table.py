import pathlib
import warnings

import numpy as np
import pandas as pd

import ecephys.utils
from ecephys import sync
from ecephys.sglx.file_mgmt import resolve_reference_probe
from ecephys.wne import constants
from ecephys.wne.constants import FileExtensions
from ecephys.wne.sglx import utils
from ecephys.wne.sglx.project import SGLXProject
from ecephys.wne.sglx.subject import SGLXSubject

_TIMEBASE = constants.PROBE_SYNC_TIMEBASE_COLUMN
_INTERPOLATE_MAX_FILE_SECS = 60


def get_session_sync_file(
    session_id: str, sglx_subject: SGLXSubject, project: SGLXProject, stream: str
) -> pathlib.Path:
    return project.get_project_subject_file(
        sglx_subject.name, f"{session_id}.prb_sync.{stream}.htsv"
    )


def do_session(
    session_id: str, sglx_subject: SGLXSubject, project: SGLXProject, stream: str = "ap"
) -> pd.DataFrame:
    """Fit one session's non-reference files onto the reference probe on within-file
    time and write ``{session_id}.prb_sync.{stream}.htsv`` to the project subject
    directory.

    The result depends only on the session's sync files, not on any experiment, so
    it is reused by every experiment containing the session. It is not applied
    directly: ``do_experiment`` converts it to the experiment-acquisition timebase.
    """
    ftab = sglx_subject.get_session_frame(session_id, ftype="bin", stream=stream)
    sync_table = get_session_sync_table(project, sglx_subject, ftab)
    ecephys.utils.write_htsv(
        sync_table, get_session_sync_file(session_id, sglx_subject, project, stream)
    )
    return sync_table


def do_experiment(
    experiment: str,
    sglx_subject: SGLXSubject,
    project: SGLXProject,
    stream: str = "ap",
    refit: bool = True,
):
    """Write ``prb_sync.{stream}.htsv`` for an experiment on the
    experiment-acquisition timebase, which is the timebase ``wne.sglx.utils``
    consumers apply it on.

    With ``refit=False`` the per-session tables already on disk are reused, so
    regrouping sessions into experiments or changing acquisition offsets does not
    reread any sync file.
    """
    experiment_ftab = sglx_subject.get_experiment_frame(
        experiment, ftype="bin", stream=stream
    )
    tables = []
    for session_id in sglx_subject.get_experiment_session_ids(experiment):
        if refit:
            session_table = do_session(session_id, sglx_subject, project, stream)
        else:
            session_table = ecephys.utils.read_htsv(
                get_session_sync_file(session_id, sglx_subject, project, stream)
            )
        tables.append(to_experiment_timebase(session_table, experiment_ftab))
    f = project.get_experiment_subject_file(
        experiment, sglx_subject.name, constants.SYNC_FNAME_MAP[stream]
    )
    ecephys.utils.write_htsv(pd.concat(tables, ignore_index=True), f)


def to_experiment_timebase(
    session_table: pd.DataFrame, experiment_ftab: pd.DataFrame
) -> pd.DataFrame:
    """Convert one session's within-file fits to the experiment-acquisition timebase.

    With ``F`` a file's ``expmtPrbAcqFirstTime``, a within-file fit
    ``t_target = slope * t_source + intercept`` becomes
    ``intercept + F_target - slope * F_source`` on experiment time; the slope is
    unchanged. Files too short to have been fitted (NaN rows) are then linearly
    interpolated between their neighbours, where intercepts are near-constant.

    Raises ``UnfinalizedRecordingError`` for a file with unknown ``F``.
    """
    found = set(session_table[_TIMEBASE].unique())
    assert found == {constants.PROBE_SYNC_SESSION_TIMEBASE}, (
        f"Expected a per-session table on {constants.PROBE_SYNC_SESSION_TIMEBASE!r} "
        f"time, found {sorted(found)}."
    )
    files = {row.path.name: row for row in experiment_ftab.itertuples()}
    table = session_table.copy()
    f_source = np.array([utils.require_acq_time(files[s]) for s in table["source"]])
    f_target = np.array([utils.require_acq_time(files[t]) for t in table["target"]])
    table["intercept"] = table["intercept"] + f_target - table["slope"] * f_source
    table[_TIMEBASE] = constants.PROBE_SYNC_TIMEBASE
    return _interpolate_short_files(table, files)


def _interpolate_short_files(table: pd.DataFrame, files: dict) -> pd.DataFrame:
    table = table.sort_values(["source_probe", "target_probe"], kind="stable")
    for probe in table["source_probe"].unique():
        is_probe = table["source_probe"] == probe
        is_na = table.loc[is_probe & table[["slope", "intercept"]].isna().any(axis=1)]
        if is_na.empty:
            continue
        if all(
            files[s].fileTimeSecs < _INTERPOLATE_MAX_FILE_SECS for s in is_na["source"]
        ):
            table.loc[is_probe, ["slope", "intercept"]] = table.loc[
                is_probe, ["slope", "intercept"]
            ].interpolate(method="linear", limit_direction="both")
    if table[["slope", "intercept"]].isna().any(axis=None):
        raise ValueError("Unable to interpolate all the missing sync info.")
    return table


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
                    _TIMEBASE: constants.PROBE_SYNC_SESSION_TIMEBASE,
                }
            )
            for file in ftab.itertuples()
        ],
        ignore_index=True,
    )


def get_session_sync_table(
    project: SGLXProject, sglx_subject: SGLXSubject, session_ftab: pd.DataFrame
) -> pd.DataFrame:
    """Fit one session's non-reference files onto the reference probe on within-file
    time (the timebase of ``*.barcodes.htsv`` and ``*.ttls.htsv``).

    A file with too few sync events gets NaN ``slope`` and ``intercept``;
    ``to_experiment_timebase`` interpolates it if it is short enough.
    """
    imSyncType = _get_session_sync_type(session_ftab)
    probes, probe_ftabs, nFiles, reference_probe = _get_probe_ftabs(session_ftab)
    fits = get_reference_session_sync_table(
        probe_ftabs[reference_probe], reference_probe
    )
    if len(probes) == 1:
        return fits
    fit_fn = {
        "barcode": _get_barcode_file_fit,
        "square_pulse": _get_square_pulse_file_fit,
        "random": _get_random_pulse_file_fit,
    }[imSyncType]
    for probe in set(probes) - {reference_probe}:
        for i in range(nFiles):
            probe_binpath = probe_ftabs[probe].iloc[i]["path"]
            reference_binpath = probe_ftabs[reference_probe].iloc[i]["path"]
            print(f"Doing file {i}: {probe_binpath.name}")
            slope, intercept = fit_fn(
                project,
                sglx_subject,
                probe,
                probe_binpath,
                reference_binpath,
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
                    _TIMEBASE: constants.PROBE_SYNC_SESSION_TIMEBASE,
                }
            )
            fits = pd.concat([fits, file_fit], ignore_index=True)
    return fits.sort_values(["source_probe", "target_probe"], kind="stable")


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
