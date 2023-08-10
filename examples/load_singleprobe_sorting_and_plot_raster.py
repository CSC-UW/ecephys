import wisc_ecephys_tools as wet
from ecephys.wne.sglx.utils import load_singleprobe_sorting

# Data
sortingProjectName = "shared_sortings"
subjectName = "CNPIX12-Santiago"
probe = "imec1"
experiment = "novel_objects_deprivation"
sorting = "sorting"
postprocessing = "postpro"
sharedDataProjectName = "shared_s3"

filters = {
    "quality": {
        "good",
        "mua",
    },  # "quality" property is "group" from phy curation. Remove noise
    "firing_rate": (0.0, float("Inf")),  # No need when plotting by depth
    # ...
}

# Plotter
aggregate_spikes_by = "depth"  # "depth"/"cluster_id" or any other property
tgt_struct_acronyms = (
    None  # Plot only target structures, in specific order. eg ["VM", "VL"]
)

### END USER

sglxSubject = wet.get_sglx_subject(subjectName)
sglxSortingProject = wet.get_sglx_project(sortingProjectName)
wneSharedProject = wet.get_wne_project(sharedDataProjectName)

si_ks_sorting = load_singleprobe_sorting(
    sglxSortingProject,
    sglxSubject,
    experiment,
    probe,
    sorting=sorting,
    postprocessing=postprocessing,
    wneAnatomyProject=wneSharedProject,
    wneHypnogramProject=wneSharedProject,
)
si_ks_sorting = si_ks_sorting.refine_clusters(
    filters,
    include_nans=True,
)

si_ks_sorting.plot_interactive_ephyviewer_raster(
    by=aggregate_spikes_by,
    tgt_struct_acronyms=tgt_struct_acronyms,
)
