import pathlib
from typing import Optional

from ecephys import hypnogram
from ecephys import units
from ecephys import utils
from ecephys.sglx import file_mgmt
from ecephys.wne import Project
from ecephys.wne.sglx import sessions
from ecephys.wne.sglx import SGLXProject
from ecephys.wne.sglx import SGLXSubject


def float_hypnogram_to_datetime(
    subj: SGLXSubject, experiment: str, hyp: hypnogram.FloatHypnogram, hyp_prb: str
) -> hypnogram.DatetimeHypnogram:
    df = hyp._df.copy()
    df["start_time"] = subj.t2dt(experiment, hyp_prb, df["start_time"])
    df["end_time"] = subj.t2dt(experiment, hyp_prb, df["end_time"])
    df["duration"] = df["end_time"] - df["start_time"]
    return hypnogram.DatetimeHypnogram(df)


def datetime_hypnogram_to_float(
    subj: SGLXSubject, experiment: str, hyp: hypnogram.DatetimeHypnogram, hyp_prb: str
) -> hypnogram.FloatHypnogram:
    df = hyp._df.copy()
    df["start_time"] = subj.dt2t(experiment, hyp_prb, df["start_time"])
    df["end_time"] = subj.dt2t(experiment, hyp_prb, df["end_time"])
    df["duration"] = df["end_time"] - df["start_time"]
    return hypnogram.FloatHypnogram(df)


def get_sglx_file_counterparts(
    project: SGLXProject,
    subject: str,
    paths: list[pathlib.Path],
    extension: str,
    remove_probe: bool = False,
    remove_stream: bool = False,
) -> list[pathlib.Path]:
    """Get counterparts to SpikeGLX raw data files.

    Counterparts are mirrored at the project's subject directory, and likely
    have different suffixes than the original raw data files.

    Parameters:
    -----------
    project_name: str
        From projects.yaml
    subject_name: str
        Subject's name within this project, i.e. subject's directory name.
    paths: list of pathlib.Path
        The raw data files to get the counterparts of.
    extension:
        The extension to replace .bin or .meta with. See `replace_ftype`.

    Returns:
    --------
    list of pathlib.Path
    """
    counterparts = sessions.mirror_raw_data_paths(
        project.get_subject_directory(subject), paths
    )  # Mirror paths at the project's subject directory
    counterparts = [
        file_mgmt.replace_ftype(p, extension, remove_probe, remove_stream)
        for p in counterparts
    ]
    return utils.remove_duplicates(counterparts)


def load_datetime_hypnogram(
    project: Project,
    experiment: str,
    subject: SGLXSubject,
    simplify: bool = True,
) -> hypnogram.DatetimeHypnogram:
    hg = project.load_float_hypnogram(experiment, subject, simplify)
    params = project.load_experiment_subject_params(experiment, subject.name)
    return float_hypnogram_to_datetime(
        subject, experiment, hg, params["hypnogram_probe"]
    )


def load_singleprobe_sorting(
    sglxSortingProject: SGLXProject,
    sglxSubject: SGLXSubject,
    experiment: str,
    alias: str,
    probe: str,
    sorting: str = "sorting",
    postprocessing: str = "postpro",
    wneAnatomyProject: Optional[Project] = None,
    allow_no_sync_file=True,
) -> units.SpikeInterfaceKilosortSorting:
    if sorting is None:
        sorting = "sorting"
    if postprocessing is None:
        postprocessing = "postpro"

    # Get function for converting SI samples to imec0 timebase
    if allow_no_sync_file:
        import warnings

        warnings.warn(
            "Maybe loading sample2time without probe-to-probe synchronization, watch out"
        )
    sample2time = sglxSortingProject.get_sample2time(
        sglxSubject.name,
        experiment,
        alias,
        probe,
        sorting,
        allow_no_sync_file=allow_no_sync_file,
    )

    # Load extractor
    extractor = sglxSortingProject.get_kilosort_extractor(
        sglxSubject.name,
        experiment,
        alias,
        probe,
        sorting,
        postprocessing=postprocessing,
    )

    # TODO: Why was this removed, and should it be restored?
    # extractor = units.si_ks_sorting.fix_isi_violations_ratio(extractor)

    # Add anatomy to the extractor, if available.
    if wneAnatomyProject is not None:
        anatomy_file = wneAnatomyProject.get_experiment_subject_file(
            experiment, sglxSubject.name, f"{probe}.structures.htsv"
        )
        assert anatomy_file.exists(), (
            f"Could not find anatomy file at: {anatomy_file}.\n"
            f"Set `wneAnatomyProject = None` in kwargs to ignore anatomy."
        )
        structs = utils.read_htsv(anatomy_file)
    else:
        structs = units.siutils.get_dummy_structure_table()
    extractor = units.siutils.add_anatomy_properties_to_extractor(extractor, structs)

    return units.SpikeInterfaceKilosortSorting(extractor, sample2time)


def load_multiprobe_sorting(
    sglxSortingProject: SGLXProject,
    sglxSubject: SGLXSubject,
    experiment: str,
    alias: str,
    probes: list[str],
    sortings: dict[str, str] = None,
    postprocessings: dict[str, str] = None,
    wneAnatomyProject: Optional[Project] = None,
    allow_no_sync_file=True,
) -> units.MultiprobeSorting:
    if sortings is None:
        sortings = {prb: None for prb in probes}
    if postprocessings is None:
        postprocessings = {prb: None for prb in probes}

    return units.MultiprobeSorting(
        {
            probe: load_singleprobe_sorting(
                sglxSortingProject,
                sglxSubject,
                experiment,
                alias,
                probe=probe,
                sorting=sortings[probe],
                postprocessing=postprocessings[probe],
                wneAnatomyProject=wneAnatomyProject,
                allow_no_sync_file=allow_no_sync_file,
            )
            for probe in probes
        }
    )
