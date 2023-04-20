import wisc_ecephys_tools as wet
from ecephys import wne
from ecephys.wne.utils import load_singleprobe_sorting

sortingProjectName = "shared_sortings"
subjectName = "CNPIX12-Santiago"
probe = "imec1"
experiment = "novel_objects_deprivation"
alias = "full"
sorting_basename = "sorting"
postprocessing_name = "postpro"
sharedDataProjectName = "shared_s3"

filters = {
    "quality": {"good", "mua"}, # "quality" property is "group" from phy curation. Remove noise
    "firing_rate": (0.5, float("Inf")),
    # ...
}

subjectsDir = wet.get_subjects_directory()
projectsFile = wet.get_projects_file()

subjLib = wne.sglx.SubjectLibrary(subjectsDir)
projLib = wne.ProjectLibrary(projectsFile)
wneSubject = subjLib.get_subject(subjectName)
wneSortingProject = projLib.get_project(sortingProjectName)
wneSharedProject = projLib.get_project(sharedDataProjectName)

si_ks_sorting = load_singleprobe_sorting(
    wneSortingProject,
    wneSubject,
    experiment,
    alias,
    probe,
    sorting_basename=sorting_basename,
    postprocessing=postprocessing_name,
    wneAnatomyProject=wneSharedProject,
    wneHypnogramProject=wneSharedProject,
)
si_ks_sorting = si_ks_sorting.refine_clusters(filters)

si_ks_sorting.plot_interactive_ephyviewer_raster()
