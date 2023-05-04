import wisc_ecephys_tools as wet
from ecephys.wne.utils import load_singleprobe_sorting
import numpy as np

# Data
sortingProjectName = "shared_sortings"
subjectName = "CNPIX11-Adrian"
experiment = "novel_objects_deprivation"
alias = "full"
probe = "imec1"
sorting = "sorting"
postprocessing = "postpro"
filters= {
    "quality": {"good", "mua"},
}, # Exclude noise

# Output Path
# Saves at /path/to/project/<experiment>/<subject>/<off_df_fname>
off_df_fname = f"off_df.{probe}.csv"
outputProjectName = "test_project"

# Params for OFF detection
tgt_states = None  # List of vigilance states over which we subset spikes. We automatically exclude NoData and bouts before/after start/end of recording
# tgt_states=["NREM"]

on_off_method = "hmmem"
on_off_params = {
    "binsize": 0.010,  # (s) (Discrete algorithm)
    "history_window_nbins": 3,  # Size of history window IN BINS
    "n_iter_EM": 200,  # Number of iterations for EM
    "n_iter_newton_ralphson": 100,
    "init_A": np.array(
        [[0.1, 0.9], [0.01, 0.99]]
    ),  # Initial transition probability matrix
    "init_state_estimate_method": "liberal",  # Method to find inital OFF states to fit GLM model with. Ignored if init_mu/alphaa/betaa are specified. Either of 'conservative'/'liberal'/'intermediate'
    "init_mu": None,  # ~ OFF rate. Fitted to data if None
    "init_alphaa": None,  # ~ difference between ON and OFF rate. Fitted to data if None
    "init_betaa": None,  # ~ Weight of recent history firing rate. Fitted to data if None,
    "gap_threshold": None,  # Merge active states separated by less than gap_threhsold
}

spatial_detection = False
spatial_params = None
# spatial_params = {
# 	# Windowing/pooling
# 	'window_size_min': 200,  # (um) Smallest spatial "grain" for pooling
# 	'window_overlap': 0.5,  # (no unit) Overlap between windows within each spatial grain
# 	'window_size_step': 200,  # (um) Increase in size of windows across successive spatial "grains"
# 	# Merging of OFF state between and across grain
# 	'merge_max_time_diff': 0.050, # (s). To be merged, off states need their start & end times to differ by less than this
# 	'nearby_off_max_time_diff': 3, # (sec). #TODO
# 	'sort_all_window_offs_by': ['off_area', 'duration', 'start_time', 'end_time'],  # How to sort all OFFs before iteratively merging
# 	'sort_all_window_offs_by_ascending': [False, False, True, True],
# }  # Only if spatial_detection = True

split_by_structure = False  # If True, run off detection separately per structure. One could perform spatial Off detection AND split by structure!

tgt_structure_acronyms = None  # List of acronyms of structures to include. All structures if None
# tgt_structure_acronyms=["Po", "VM"]

n_jobs = 10  # Only for spatial OFF detection

###

savepath = wet.get_wne_project(outputProjectName).get_experiment_subject_file(experiment, subjectName, off_df_fname)
print(f"\nWill save OFF at {savepath}\n")

wneSubject = wet.get_wne_subject(subjectName)
sortingProject = wet.get_wne_project(sortingProjectName)
anatomyProject = wet.get_wne_project("shared_s3")
hypnogramProject = wet.get_wne_project("shared_s3")

sorting = load_singleprobe_sorting(
    sortingProject,
    wneSubject,
    experiment,
    alias,
    probe,
    sorting,
    postprocessing,
    wneAnatomyProject=anatomyProject,
    wneHypnogramProject=hypnogramProject,
)
sorting = sorting.refine_clusters(
    filters={
        "quality": {"good", "mua"},
    }, # Exclude noise
    include_nans=True,
)

off_df = sorting.run_off_detection(
    tgt_states=tgt_states,
    on_off_method=on_off_method,
    on_off_params=on_off_params,
    spatial_detection=spatial_detection,
    spatial_params=spatial_params,
    split_by_structure=split_by_structure,
    tgt_structure_acronyms=tgt_structure_acronyms,
    n_jobs=n_jobs,
)
off_df.to_csv(savepath)
