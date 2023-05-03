import wisc_ecephys_tools as wet
from ecephys import wne
from ecephys.utils import read_htsv
import pandas as pd

sharedDataProjectName = "shared_s3"
subjectName = "ANPIX33-Arvid"
experiment = "discoflow-day1"

wneSubject = wne.sglx.SubjectLibrary(wet.get_subjects_directory()).get_subject(subjectName)
wneSharedProject = wne.ProjectLibrary(wet.get_projects_file()).get_project(sharedDataProjectName)

fpath = wneSharedProject.get_experiment_subject_file(
    experiment=experiment, 
    subject=subjectName,
    fname="ephys_stimulus_times.csv",
)
print(pd.read_csv(fpath))