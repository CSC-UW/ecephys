import wisc_ecephys_tools as wet
from ecephys import wne
from ecephys.wne.utils import load_multiprobe_sorting

# Data
sortingProjectName = "shared_sortings"
subjectName = "CNPIX12-Santiago"
experiment = "novel_objects_deprivation"
alias = "full"
probes = [
    "imec1",
]
sortings = {
    "imec1": "sorting",

}
postprocessings = {
    "imec1": "postpro",
}
sharedDataProjectName = "shared_s3"

df_filters = {
    "quality": {"good", "mua"}, # "quality" property is "group" from phy curation. Remove noise
    "firing_rate": (0.5, float("Inf")),
    # ...
}
filters = {
    "imec1": df_filters
} # Probe-specific

# Plotter
aggregate_spikes_by = "depth"  # "depth"/"cluster_id" or any other property
tgt_struct_acronyms = {
    "imec1": None # Plot only target structures, in specific order. eg ["VM", "VL"]
}

### END USER

subjectsDir = wet.get_subjects_directory()
projectsFile = wet.get_projects_file()

subjLib = wne.sglx.SubjectLibrary(subjectsDir)
projLib = wne.ProjectLibrary(projectsFile)
wneSubject = subjLib.get_subject(subjectName)
wneSortingProject = projLib.get_project(sortingProjectName)
wneSharedProject = projLib.get_project(sharedDataProjectName)

multiprobe_sorting = load_multiprobe_sorting(
    wneSortingProject,
    wneSubject,
    experiment,
    alias,
    probes,
    sortings=sortings,
    postprocessings=postprocessings,
    wneAnatomyProject=wneSharedProject,
    wneHypnogramProject=wneSharedProject,
)
multiprobe_sorting = multiprobe_sorting.refine_clusters(filters)

multiprobe_sorting.plot_interactive_ephyviewer_raster(
    by=aggregate_spikes_by,
    tgt_struct_acronyms=tgt_struct_acronyms,
)