from enum import StrEnum
from typing import Final

# Visbrain will automatically resample everything to 100Hz, so just nip it in the bud.
VISBRAIN_FS: Final = 100


class FileExtensions(StrEnum):
    BARCODE = ".barcodes.htsv"
    TTL = ".ttls.htsv"
    NETCDF = ".nc"
    EMG = ".emg.nc"
    EDF = ".edf"
    BDF = ".bdf"
    LFP = ".lf.zarr"
    ARTIFACTS = ".artifacts.csv"
    VISBRAIN = ".hypnogram.txt"


class Files(StrEnum):
    AP_SYNC = "prb_sync.ap.htsv"
    LF_SYNC = "prb_sync.lf.htsv"
    EXP_PARAMS = "experiment_params.json"
    EMG = "emg.nc"
    ARTIFACTS = "artifacts.htsv"
    HYPNOGRAM = "hypnogram.htsv"
    DATETIME_HYPNOGRAM = "hypnogram_datetime.htsv"
    HIPPOCAMPAL_SUBREGIONS = "hippocampal_subregions.json"  # TODO: Not general. Remove to project-specific repositories.
    SCORING_LFP = "scoring_lfp.zarr"
    SCORING_EMG = "scoring_emg.zarr"
    SCORING_BDF = "scoring_signals.bdf"


SYNC_FNAME_MAP = {"ap": Files.AP_SYNC, "lf": Files.LF_SYNC}

#  TODO: The following are not constants, and should be elsewhere.
SIMPLIFIED_ARTIFACTS = {
    "unlabeled": "Artifact",
    "artifact": "Artifact",
    "flat": "Artifact",
    "scrambled": "Artifact",
}  # "type" column in consolidated artifacts

SIMPLIFIED_STATES = {
    "Wake": "Wake",
    "W": "Wake",
    "aWk": "Wake",
    "qWk": "Wake",
    "QWK": "Wake",
    "Arousal": "MA",
    "MA": "MA",
    "Trans": "Other",
    "NREM": "NREM",
    "N1": "NREM",
    "N2": "NREM",
    "IS": "IS",
    "REM": "REM",
    "Art": "Artifact",
    "None": "Other",
    # Mice hypnograms:
    "Brief-Arousal": "MA",
    "Transition-to-Wake": "Other",
    "Transition-to-NREM": "Other",
    "Transition-to-REM": "Other",  # TODO: Possibly equivalent to IS
    "Wake-Good": "Wake",
}
