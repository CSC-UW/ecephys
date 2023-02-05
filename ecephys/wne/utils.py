from typing import Optional
from .. import utils, units
from .projects import Project
from . import sglx


def load_singleprobe_sorting(
    wneSortingProject: Project,
    wneSubject: sglx.Subject,
    experiment: str,
    alias: str,
    probe: str,
    sorting: str,
    wneAnatomyProject: Optional[Project] = None,
) -> units.SpikeInterfaceKilosortSorting:

    # Get function for converting SI samples to imec0 timebase
    sample2time = wneSortingProject.get_sample2time(
        wneSubject, experiment, alias, probe, sorting
    )

    # Load extractor
    extractor = wneSortingProject.get_kilosort_extractor(
        wneSubject.name, experiment, alias, probe, sorting
    )

    # Fix extractor
    extractor = units.fix_isi_violations_ratio(extractor)
    extractor = units.fix_noise_cluster_labels(extractor)
    extractor = units.fix_uncurated_cluster_labels(extractor)

    # Add anatomy to the extractor, if available.
    if wneAnatomyProject is None:
        wneAnatomyProject = wneSortingProject
    anatomy_file = wneAnatomyProject.get_experiment_subject_file(
        experiment, wneSubject.name, f"{probe}.structures.htsv"
    )
    if anatomy_file.exists():
        structs = utils.read_htsv(anatomy_file)
        extractor = units.add_cluster_structures(extractor, structs)

    return units.SpikeInterfaceKilosortSorting(extractor, sample2time)
