import wisc_ecephys_tools as wet
from ecephys.wne.sglx.utils import load_multiprobe_sorting

# Data
sortingProjectName = "shared_sortings"
subjectName = "CNPIX12-Santiago"
experiment = "novel_objects_deprivation"
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
    "firing_rate": (0.0, float("Inf")),  # No need when plotting by depth
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

sglxSubject = wet.get_sglx_subject(subjectName)
sglxSortingProject = wet.get_sglx_project(sortingProjectName)
wneSharedProject = wet.get_wne_project(sharedDataProjectName)

multiprobe_sorting = load_multiprobe_sorting(
    sglxSortingProject,
    sglxSubject,
    experiment,
    probes,
    sortings=sortings,
    postprocessings=postprocessings,
    wneAnatomyProject=wneSharedProject,
    wneHypnogramProject=wneSharedProject,
)
multiprobe_sorting = multiprobe_sorting.refine_clusters(
    filters,
    include_nans=True,
)

multiprobe_sorting.plot_interactive_ephyviewer_raster(
    by=aggregate_spikes_by,
    tgt_struct_acronyms=tgt_struct_acronyms,
)