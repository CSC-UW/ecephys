import wisc_ecephys_tools as wet
import ecephys.utils
from ecephys.wne.sglx.utils import load_singleprobe_sorting
import numpy as np
import argparse

# Parse experiment, alias, subjectName, probe from command line
example_text = """
example:

python run_off_detection.py experiment alias CNPIX4-Doppio,imec0
"""
parser = argparse.ArgumentParser(
    description=(f"Run off detection."),
    epilog=example_text,
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument("experiment", type=str, help="Name of experiment we run off for.")
parser.add_argument(
    "alias", type=str, help="Name of alias we run off for."
)  # TODO: Remove? Not used?
parser.add_argument(
    "subject_probe",
    type=str,
    help="Target probe. Comma-separated string of the form `'<subjectName>,<probe>'",
)
args = parser.parse_args()

experiment = args.experiment
alias = args.alias
subject_probe = args.subject_probe
subjectName, probe = subject_probe.split(",")

# Data
sortingProjectName = "shared_sortings"
sorting = None
postprocessing = None
filters = {
    "quality": {"good", "mua"},
}  # Exclude noise

# Params for OFF detection
# tgt_states = None  # List of vigilance states over which we subset spikes. We automatically exclude NoData and bouts before/after start/end of recording
tgt_states = ["NREM", "Wake", "REM"]
split_by_state = True

on_off_method = "hmmem"
on_off_params = {
    "binsize": 0.010,  # (s) (Discrete algorithm)
    "history_window_nbins": 3,  # Size of history window IN BINS
    "n_iter_EM": 200,  # Number of iterations for EM
    "n_iter_newton_ralphson": 100,
    "init_A": np.array(
        [[0.1, 0.9], [0.01, 0.99]]
    ),  # Initial transition probability matrix
    # "init_state_estimate_method": "liberal",  # Method to find inital OFF states to fit GLM model with. Ignored if init_mu/alphaa/betaa are specified. Either of 'conservative'/'liberal'/'intermediate'
    "init_state_estimate_method": "conservative",  # Method to find inital OFF states to fit GLM model with. Ignored if init_mu/alphaa/betaa are specified. Either of 'conservative'/'liberal'/'intermediate'
    "init_mu": None,  # ~ OFF rate. Fitted to data if None
    "init_alphaa": None,  # ~ difference between ON and OFF rate. Fitted to data if None
    "init_betaa": None,  # ~ Weight of recent history firing rate. Fitted to data if None,
    "gap_threshold": 0.050,  # Merge active states separated by less than gap_threhsold
}

spatial_detection = False
# spatial_detection = True
spatial_params = None
# spatial_params = {
#     # Windowing/pooling
#     'window_size_min': 200,  # (um) Smallest spatial "grain" for pooling
#     'window_overlap': 0.5,  # (no unit) Overlap between windows within each spatial grain
#     'window_size_step': 200,  # (um) Increase in size of windows across successive spatial "grains"
#     'window_min_fr': 10, # (Hz) Minimum aggregate FR within window to include it
#     # Merging of OFF state between and across grain
#     'merge_max_time_diff': 0.050, # (s). To be merged, off states need their start & end times to differ by less than this
#     'nearby_off_max_time_diff': 3, # (sec). #TODO
#     'sort_all_window_offs_by': ['off_area', 'duration', 'start_time', 'end_time'],  # How to sort all OFFs before iteratively merging
#     'sort_all_window_offs_by_ascending': [False, False, True, True],
# }  # Only if spatial_detection = True

# tgt_structure_acronyms=["Po", "VM"]
tgt_structure_acronyms = (
    None  # List of acronyms of structures to include. All structures if None
)
# structure_acronyms_to_ignore = []
structure_acronyms_to_ignore = [
    "CLA",
    "HY",
    "NAc",
    "NAc-c",
    "NAc-sh",
    "SN",
    "V",
    "ZI",
    "ZI-cBRF",
    "cc-ec-cing-dwn",
    "cfp",
    "eml",
    "ic-cp-lfp-py",
    "ml",
    "wmt",
]

n_jobs = 10  # Only for spatial OFF detection


# Output Path
# Saves at /path/to/project/<experiment>/<subject>/offs/<off_df_fname>
# eg: "imec0.Po.global_offs_bystate_conservative_0.050.htsv"
# Load with wneProject.load_offs_df()
def get_off_df_fname(acronym=None):
    fname = f"{probe}."
    if acronym is not None:
        fname += f"{acronym}."
    fname += f"{'spatial' if spatial_detection else 'global'}_offs"
    fname += f"{'_bystate' if split_by_state else ''}"
    fname += f"_{on_off_params['init_state_estimate_method']}"
    fname += f"_{on_off_params['gap_threshold']}"
    fname += ".htsv"
    return fname


outputProjectName = "shared_s3"

###

off_dirpath = (
    wet.get_wne_project(outputProjectName).get_experiment_subject_directory(
        experiment,
        subjectName,
    )
    / "offs"
)
off_dirpath.mkdir(exist_ok=True, parents=True)

print(f"\nWill save OFFs at eg {off_dirpath/get_off_df_fname('<acronym>')}\n")

sglxSubject = wet.get_sglx_subject(subjectName)
sortingProject = wet.get_sglx_project(sortingProjectName)
anatomyProject = wet.get_wne_project("shared_s3")
hypnogramProject = wet.get_wne_project("shared_s3")

sorting = load_singleprobe_sorting(
    sortingProject,
    sglxSubject,
    experiment,
    probe,
    alias=alias,
    sorting=sorting,
    postprocessing=postprocessing,
    wneAnatomyProject=anatomyProject,
)
sorting = sorting.refine_clusters(
    simple_filters=filters,  # Exclude noise
    callable_filters=None,
    include_nans=True,
)

hg = hypnogramProject.load_float_hypnogram(experiment, sglxSubject.name, simplify=True)

# Remove structures to ignore
all_structures = sorting.structures_by_depth
sorting = sorting.select_structures(
    [s for s in all_structures if not s in structure_acronyms_to_ignore]
)
print("Structures to run:", sorting.structures_by_depth)

for acronym in sorting.structures_by_depth:
    structure_sorting = sorting.select_structures([acronym])

    off_df = structure_sorting.run_off_detection(
        hg,
        tgt_states=tgt_states,
        split_by_state=split_by_state,
        on_off_method=on_off_method,
        on_off_params=on_off_params,
        spatial_detection=spatial_detection,
        spatial_params=spatial_params,
        n_jobs=n_jobs,
    )
    off_df["structure"] = acronym
    off_df["acronym"] = acronym

    savepath = off_dirpath / get_off_df_fname(acronym)
    print(f"Save aggregate off frame at {savepath}")

    if len(off_df):
        ecephys.utils.write_htsv(off_df, savepath)
