import wisc_ecephys_tools as wet
from ecephys import wne
from ecephys.wne.sglx.pipeline.sorting_pipeline import \
    SpikeInterfaceSortingPipeline

projectName = "my_project"
subjectName = "CNPIX15-Claude"
probe = "imec0"
experiment = "novel_objects_deprivation"
alias = "recovery_sleep"
sorting_basename = "sorting_df"


subjectsDir = wet.get_subjects_directory()
projectsFile = wet.get_projects_file()

subjLib = wne.sglx.SubjectLibrary(subjectsDir)
projLib = wne.ProjectLibrary(projectsFile)
wneSubject = subjLib.get_subject(subjectName)
wneProject = projLib.get_project(projectName)

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