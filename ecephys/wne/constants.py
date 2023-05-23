NETCDF_EXT = ".nc"
EMG_EXT = ".emg.nc"
EDF_EXT = ".edf"
BDF_EXT = ".bdf"
LFP_EXT = ".lf.zarr"
ARTIFACTS_EXT = ".artifacts.csv"
VISBRAIN_EXT = ".hypnogram.txt"
VISBRAIN_FS = 100  # Visbrain will automatically resample everything to 100Hz, so just nip it in the bud.

EXP_PARAMS_FNAME = "experiment_params.json"
SORTING_PIPELINE_PARAMS_FNAME = "sorting_pipeline_params.yaml"
EMG_FNAME = "emg.nc"
ARTIFACTS_FNAME = "artifacts.htsv"
HYPNOGRAM_FNAME = "hypnogram.htsv"
DATETIME_HYPNOGRAM_FNAME = "hypnogram_datetime.htsv"

SCORING_LFP = "scoring_lfp.zarr"
SCORING_EMG = "scoring_emg.zarr"
SCORING_BDF = "scoring_signals.bdf"

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
}
