import wisc_ecephys_tools as wet
from ecephys.utils import read_htsv
import pandas as pd

sharedDataProjectName = "shared_s3"
subjectName = "ANPIX33-Arvid"
experiment = "discoflow-day1"

wneSharedProject = wet.get_wne_project(sharedDataProjectName)

fpath = wneSharedProject.get_experiment_subject_file(
    experiment=experiment, 
    subject=subjectName,
    fname="ephys_stimulus_times.csv",
)
print(pd.read_csv(fpath))