"""Regression tests for handling un-finalized (crashed) SpikeGLX recordings whose
.meta lost the close-time field ``firstSample``.

See ecephys.sglx.repair_metadata and the cache-build path in
ecephys.sglx.file_mgmt.filelist_to_frame.
"""

import types
from pathlib import Path

import numpy as np
import pytest

from ecephys.sglx import file_mgmt
from ecephys.wne.sglx.utils import UnfinalizedRecordingError, require_acq_time

# A real SpikeGLX 3B2 .meta (Quincy 09-14 imec0.ap), minus the three giant
# ~imroTbl / ~snsChanMap / ~snsShankMap lines, which the file-table parse path
# does not consult. `firstSample` here is large and nonzero (62334444), as is
# typical -- it is NOT 0 for a first-trigger file.
HEALTHY_META = """\
acqApLfSy=384,384,1
appVersion=20201103
fileCreateTime=2022-09-14T18:00:45
fileName=E://9-14-2022_g0/9-14-2022_g0_imec0/9-14-2022_g0_t0.imec0.ap.bin
fileSHA1=7A767E9C3301CCF9DFE34F0DBA5E8289DD492878
fileSizeBytes=324722049960
fileTimeSecs=14057.207516723509
firstSample=62334444
gateMode=Immediate
imAiRangeMax=0.6
imAiRangeMin=-0.6
imCalibrated=true
imDatApi=3.31
imRoFile=C:/Users/UWisc/Desktop/tip_ref.imro
imSampRate=30000.051397
imStdby=
imTrgRising=true
imTrgSource=0
nDataDirs=1
nSavedChans=385
snsApLfSy=384,0,1
snsSaveChanSubset=0:383,768
typeThis=imec
userNotes=
"""

N_SAVED_CHANS = 385
FILE_SIZE_BYTES = 324722049960
SAMP_RATE = 30000.051397
EXPECTED_NFILESAMP = FILE_SIZE_BYTES / (2 * N_SAVED_CHANS)


def _make_bin(tmp_path: Path, meta_text: str) -> Path:
    """Write a SGLX-named .bin (empty) + sibling .meta; return the .bin path."""
    binpath = tmp_path / "test_g0_t0.imec0.ap.bin"
    binpath.touch()  # filelist_to_frame reads fileSizeBytes from meta, not disk
    binpath.with_suffix(".meta").write_text(meta_text)
    return binpath


def test_healthy_meta_unchanged(tmp_path):
    """A finalized recording is parsed exactly as before (no-op guard path)."""
    binpath = _make_bin(tmp_path, HEALTHY_META)
    frame = file_mgmt.filelist_to_frame([binpath])
    assert len(frame) == 1
    row = frame.iloc[0]
    assert row["firstSample"] == 62334444
    assert frame["firstSample"].dtype.kind in ("i", "u")  # stays integer
    assert not np.isnan(row["firstTime"])
    assert row["nFileSamp"] == EXPECTED_NFILESAMP


def test_missing_firstSample_yields_nan_not_keyerror(tmp_path):
    """An un-finalized recording (no firstSample) loads with NaN absolute times
    and valid nFileSamp/fileDuration, instead of raising KeyError."""
    unfinalized = "".join(
        line + "\n"
        for line in HEALTHY_META.splitlines()
        if not line.startswith("firstSample=")
    )
    binpath = _make_bin(tmp_path, unfinalized)

    frame = file_mgmt.filelist_to_frame([binpath])  # must not raise
    assert len(frame) == 1
    row = frame.iloc[0]
    # Absolute, firstSample-derived columns are NaN (unknown):
    assert np.isnan(row["firstSample"])
    assert np.isnan(row["lastSample"])
    assert np.isnan(row["firstTime"])
    assert np.isnan(row["lastTime"])
    # Length/duration are recoverable and correct:
    assert row["nFileSamp"] == EXPECTED_NFILESAMP
    assert row["fileDuration"] == pytest.approx(EXPECTED_NFILESAMP / SAMP_RATE)


def test_require_acq_time():
    """Pass-through when known; clear error when unknown (NaN)."""
    known = types.SimpleNamespace(expmtPrbAcqFirstTime=12.5, path=Path("a.bin"))
    assert require_acq_time(known) == 12.5

    unknown = types.SimpleNamespace(expmtPrbAcqFirstTime=np.nan, path=Path("b.bin"))
    with pytest.raises(UnfinalizedRecordingError):
        require_acq_time(unknown)
