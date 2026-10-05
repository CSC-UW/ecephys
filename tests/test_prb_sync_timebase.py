"""Probe sync tables are fitted per session on within-file time, converted to the
experiment-acquisition timebase per experiment, and applied on that timebase.

Every barcode seen by both a non-reference probe and the reference probe is one
event. Its time on each probe is known in within-file seconds (``.barcodes.htsv``
``time``) and in experiment-acquisition seconds (plus ``expmtPrbAcqFirstTime``).
After mapping the non-reference time through the experiment table with
``get_time2time`` it must land on the reference probe's time.
"""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

import ecephys.utils
from ecephys.wne import constants
from ecephys.wne.constants import FileExtensions
from ecephys.wne.sglx import utils as sglx_utils
from ecephys.wne.sglx.pipeline import generate_probe_sync_table

SCRATCH = Path("/Volumes/scratch/neuropixels/ecephys/prb_sync_timebase/test_tables")
CASES = [
    ("ANPIX39-Roland", "discoflow-day1"),
    ("CNPIX18-Pier", "novel_objects_deprivation"),
]
RESIDUAL_S = 0.5e-3
TB = constants.PROBE_SYNC_TIMEBASE_COLUMN


def _table(**cols) -> pd.DataFrame:
    base = {
        "source": ["a_g0_t0.imec1.ap.bin"],
        "target": ["a_g0_t0.imec0.ap.bin"],
        "source_probe": ["imec1"],
        "target_probe": ["imec0"],
        "slope": [1.0],
        "intercept": [0.0],
    }
    return pd.DataFrame({**base, **cols})


def test_loader_refuses_table_without_timebase(tmp_path):
    f = tmp_path / "prb_sync.ap.htsv"
    ecephys.utils.write_htsv(_table(), f)
    with pytest.raises(ValueError, match="timebase"):
        sglx_utils.load_probe_sync_table(f)


def test_loader_refuses_session_timebase(tmp_path):
    f = tmp_path / "prb_sync.ap.htsv"
    ecephys.utils.write_htsv(_table(**{TB: [constants.PROBE_SYNC_SESSION_TIMEBASE]}), f)
    with pytest.raises(ValueError, match=constants.PROBE_SYNC_SESSION_TIMEBASE):
        sglx_utils.load_probe_sync_table(f)


def test_loader_accepts_experiment_timebase(tmp_path):
    f = tmp_path / "prb_sync.ap.htsv"
    ecephys.utils.write_htsv(_table(**{TB: [constants.PROBE_SYNC_TIMEBASE]}), f)
    assert len(sglx_utils.load_probe_sync_table(f)) == 1


def _ftab(files: dict[str, float], file_secs: float = 7200.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "path": [Path("/x") / name for name in files],
            "expmtPrbAcqFirstTime": list(files.values()),
            "fileTimeSecs": file_secs,
        }
    )


def _session_table(trigs, slope, intercept) -> pd.DataFrame:
    rows = []
    for t in trigs:
        rows.append(
            ("imec0", 1.0, 0.0, f"a_g0_t{t}.imec0.ap.bin", f"a_g0_t{t}.imec0.ap.bin")
        )
        rows.append(
            (
                "imec1",
                slope,
                intercept,
                f"a_g0_t{t}.imec1.ap.bin",
                f"a_g0_t{t}.imec0.ap.bin",
            )
        )
    df = pd.DataFrame(
        rows, columns=["source_probe", "slope", "intercept", "source", "target"]
    )
    df["target_probe"] = "imec0"
    df[TB] = constants.PROBE_SYNC_SESSION_TIMEBASE
    return df


def test_to_experiment_timebase_shifts_intercept_only():
    slope, c = 1 + 2e-6, 1.7e-3
    f_src, f_tgt = 80_000.0, 80_000.02
    ftab = _ftab({"a_g0_t0.imec0.ap.bin": f_tgt, "a_g0_t0.imec1.ap.bin": f_src})
    out = generate_probe_sync_table.to_experiment_timebase(
        _session_table([0], slope, c), ftab
    )
    assert set(out[TB]) == {constants.PROBE_SYNC_TIMEBASE}
    ref = out[out["source_probe"] == "imec0"].iloc[0]
    assert (ref["slope"], ref["intercept"]) == (1.0, 0.0)
    prb = out[out["source_probe"] == "imec1"].iloc[0]
    assert prb["slope"] == slope
    assert prb["intercept"] == pytest.approx(c + f_tgt - slope * f_src, abs=1e-12)
    t_within = 1234.5
    t_ref_within = slope * t_within + c
    assert prb["slope"] * (t_within + f_src) + prb["intercept"] == pytest.approx(
        t_ref_within + f_tgt, abs=1e-9
    )


def test_to_experiment_timebase_refuses_converted_input():
    ftab = _ftab({"a_g0_t0.imec0.ap.bin": 0.0, "a_g0_t0.imec1.ap.bin": 0.0})
    table = _session_table([0], 1.0, 0.0)
    table[TB] = constants.PROBE_SYNC_TIMEBASE
    with pytest.raises(AssertionError, match=constants.PROBE_SYNC_SESSION_TIMEBASE):
        generate_probe_sync_table.to_experiment_timebase(table, ftab)


def test_to_experiment_timebase_raises_on_unfinalized_file():
    ftab = _ftab({"a_g0_t0.imec0.ap.bin": 0.0, "a_g0_t0.imec1.ap.bin": np.nan})
    with pytest.raises(sglx_utils.UnfinalizedRecordingError):
        generate_probe_sync_table.to_experiment_timebase(
            _session_table([0], 1.0, 0.0), ftab
        )


def test_to_experiment_timebase_interpolates_short_unfitted_file():
    table = _session_table([0, 1, 2], 1.0, 0.0)
    table.loc[
        (table["source_probe"] == "imec1") & (table["source"].str.contains("_t1.")),
        ["slope", "intercept"],
    ] = np.nan
    files = {f"a_g0_t{t}.imec{p}.ap.bin": 7200.0 * t for t in range(3) for p in (0, 1)}
    ftab = _ftab(files)
    ftab.loc[ftab["path"].map(lambda p: "_t1." in p.name), "fileTimeSecs"] = 30.0
    out = generate_probe_sync_table.to_experiment_timebase(table, ftab)
    assert not out[["slope", "intercept"]].isna().any(axis=None)
    ftab["fileTimeSecs"] = 7200.0
    with pytest.raises(ValueError, match="interpolate"):
        generate_probe_sync_table.to_experiment_timebase(table, ftab)


def _barcodes(project, subject, path) -> pd.DataFrame:
    [f] = sglx_utils.get_sglx_file_counterparts(
        project, subject, [path], FileExtensions.BARCODE
    )
    return ecephys.utils.read_htsv(f)


@pytest.mark.requires_nfs
@pytest.mark.parametrize("subject,experiment", CASES)
def test_shared_barcodes_agree_after_mapping(subject, experiment):
    wet = pytest.importorskip("wisc_ecephys_tools")
    if not Path("/Volumes/npx_nfs/shared_s3").is_dir():
        pytest.skip("npx_nfs not mounted")
    project = wet.get_sglx_project("shared")
    sglx_subject = wet.get_sglx_subject(subject)
    ftab = sglx_subject.get_ap_bin_table(experiment)

    tables = []
    for id in sglx_subject.get_experiment_session_ids(experiment):
        session_table = generate_probe_sync_table.get_session_sync_table(
            project,
            sglx_subject,
            sglx_subject.get_session_frame(id, ftype="bin", stream="ap"),
        )
        assert set(session_table[TB]) == {constants.PROBE_SYNC_SESSION_TIMEBASE}
        ecephys.utils.write_htsv(
            session_table, SCRATCH / f"{subject}.{id}.prb_sync.ap.htsv"
        )
        tables.append(
            generate_probe_sync_table.to_experiment_timebase(session_table, ftab)
        )
    out = SCRATCH / f"{subject}.{experiment}.{constants.Files.AP_SYNC}"
    ecephys.utils.write_htsv(pd.concat(tables, ignore_index=True), out)
    table = sglx_utils.load_probe_sync_table(out)

    production = project.get_experiment_subject_file(
        experiment, subject, constants.Files.AP_SYNC
    )
    if TB not in ecephys.utils.read_htsv(production):
        with pytest.raises(ValueError, match="timebase"):
            sglx_utils.load_probe_sync_table(production)

    by_name = {r.path.name: r for r in ftab.itertuples()}
    targets = table.set_index("source")["target"]
    for probe in set(ftab["probe"]) - set(
        table.loc[table["slope"] == 1.0, "source_probe"]
    ):
        probe_ftab = ftab[ftab["probe"] == probe].reset_index(drop=True)
        t2t = sglx_utils.get_time2time(table, probe_ftab)
        for f in probe_ftab.itertuples():
            f0 = by_name[targets[f.path.name]]
            shared = pd.merge(
                _barcodes(project, subject, f.path),
                _barcodes(project, subject, f0.path),
                on="value",
                suffixes=("_probe", "_ref"),
            )
            mapped = t2t(shared["time_probe"].values + f.expmtPrbAcqFirstTime)
            err = mapped - (shared["time_ref"].values + f0.expmtPrbAcqFirstTime)
            assert abs(np.median(err)) < RESIDUAL_S, (
                probe,
                f.path.name,
                np.median(err),
            )
