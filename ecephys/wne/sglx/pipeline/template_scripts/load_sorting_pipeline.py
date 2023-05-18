import wisc_ecephys_tools as wet
from ecephys.wne.sglx.pipeline.sorting_pipeline import SpikeInterfaceSortingPipeline

projectName = "my_project"
subjectName = "CNPIX15-Claude"
probe = "imec0"
experiment = "novel_objects_deprivation"
alias = "recovery_sleep"
sorting_basename = "sorting_df"

wneSubject = wet.get_wne_subject(subjectName)
wneProject = wet.get_wne_project(projectName)

sorting_pipeline = SpikeInterfaceSortingPipeline.load_from_folder(
    wneProject,
    wneSubject,
    experiment,
    alias,
    probe,
    sorting_basename,
)
sorting_pipeline.get_raw_si_recording()
print(sorting_pipeline)
