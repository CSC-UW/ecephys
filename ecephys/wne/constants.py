NETCDF_EXT = ".nc"
EMG_EXT = ".emg.nc"
EDF_EXT = ".edf"
ARTIFACTS_EXT = ".artifacts.csv"
VISBRAIN_EXT = ".hypnogram.txt"

EXP_PARAMS_FNAME = "experiment_params.json"
SORTING_PIPELINE_PARAMS_FNAME = "sorting_pipeline_params.yaml"
EMG_FNAME = "emg.nc"
ARTIFACTS_FNAME = "artifacts.htsv"
HYPNOGRAM_FNAME = "hypnogram.htsv"
DATETIME_HYPNOGRAM_FNAME = "hypnogram_datetime.htsv"

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
