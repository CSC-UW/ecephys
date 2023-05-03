from typing import Optional

import numpy as np

from ecephys import hypnogram, units, utils
from ecephys.wne import sglx
from ecephys.wne.projects import Project
from ecephys.utils.pdutils import get_edges_start_end_samples_df


def load_hypnogram_for_si_slicing(
    wneHypnoProject: Project,
    wneSortingProject: Project,
    wneSubject,
    experiment: str,
    alias: str,
    probe: str,
    sorting: str,
    precision_s: float = None,
    allow_no_sync_file: bool = False,
    simplify: bool = True,
) -> hypnogram.Hypnogram:
    """Return df with `state`, `start_frame`, `end_frame` columns.

    Because of the possible gaps or overlap between concatenated sglx files,
    getting from timestamped epochs to corresponding start and end samples in
    the sorting is not trivial.

    Here we label each sample with the corresponding hypnogram state,
    and then find the edges of the array of states.

    The returned hypnogram has `start_frame` and `end_frame` columns
    which can be used to slice the corresponding sorting/recording.

    Parameters:
    ===========
    wneHypnoProject: Project
        Used to pull whole experiment hypnogram
    wneSortingProject: Project,
        Used to pull segment file and sample2time function of interest
    wneSubject: Subject,
    experiment: str,
    alias: str,
    probe: str,
    sorting: str,
    precision_s: float,
        Used to downsample the sample vector for speed.
    allow_no_sync_file: bool,
        Passed to sample2time function
    simplify: bool,
        Simplify hypnogram states.

    Returns:
    ========
    hypnogram.Hypnogram: Hypnogram with 'start_time', 'end_time', 'duration' columns
        in expmtAcqPrb timebase, and `start_frame`, `end_frame` columns for each
        bout matching the spikeinterface sorting/recording frame ids.
    """
    raw_hypno = wneHypnoProject.load_hypnogram(
        experiment, wneSubject.name, simplify=simplify
    )

    segments = wneSortingProject.load_segments_table(
        wneSubject,
        experiment,
        alias,
        probe,
        sorting,
        return_all_segment_types=False,
    )
    sample2time = wneSortingProject.get_sample2time(
        wneSubject,
        experiment,
        alias,
        probe,
        sorting,
        allow_no_sync_file=allow_no_sync_file,
    )

    if precision_s is None:
        decimation_n_samples = 1
    else:
        decimation_n_samples = int(precision_s * segments["imSampRate"].values[0])
    assert decimation_n_samples >= 1

    total_n_samples = segments.nSegmentSamp.sum()
    decimated_frame_ids = np.arange(0, total_n_samples, decimation_n_samples)
    decimated_frame_timestamps = sample2time(decimated_frame_ids)

    # To get from timestamp vector to state vector, we split
    # by segment because Hypnogram.get_states() assumes monotonocity
    # of time vector
    segment_decimated_frame_states_list = []
    for segment in segments.itertuples():
        segment_start_time = segment.segmentExpmtPrbAcqFirstTime
        segment_end_time = segment.segmentExpmtPrbAcqLastTime
        segment_decimated_frame_timestamps = decimated_frame_timestamps[
            (segment_start_time <= decimated_frame_timestamps)
            & (decimated_frame_timestamps <= segment_end_time)
        ]
        segment_decimated_frame_states_list.append(
            raw_hypno.get_states(
                segment_decimated_frame_timestamps, default_value="Other"
            )
        )

    # (nsamples,) array of states for whole sorting
    decimated_frame_states = np.concatenate(segment_decimated_frame_states_list)

    # df with start_frame, end_frame columns, still in decimated indices
    decimated_frame_hypno = get_edges_start_end_samples_df(decimated_frame_states)

    # df with start_frame, end_frame columns, in original indices
    frame_hypno = decimated_frame_hypno.copy()
    frame_hypno["start_frame"] = (
        decimation_n_samples * decimated_frame_hypno["start_frame"]
    ).clip(upper=total_n_samples - 1)
    frame_hypno["end_frame"] = (
        decimation_n_samples * decimated_frame_hypno["end_frame"]
    ).clip(upper=total_n_samples - 1)
    frame_hypno["start_time"] = sample2time(frame_hypno["start_frame"])
    frame_hypno["end_time"] = sample2time(frame_hypno["end_frame"])
    frame_hypno["duration"] = frame_hypno["end_time"] - frame_hypno["start_time"]

    return hypnogram.Hypnogram(frame_hypno)


def load_singleprobe_sorting(
    wneSortingProject: Project,
    wneSubject: sglx.Subject,
    experiment: str,
    alias: str,
    probe: str,
    sorting: str = "sorting",
    postprocessing: str = "postpro",
    wneAnatomyProject: Optional[Project] = None,
    wneHypnogramProject: Optional[Project] = None,
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
    sample2time = wneSortingProject.get_sample2time(
        wneSubject,
        experiment,
        alias,
        probe,
        sorting,
        allow_no_sync_file=allow_no_sync_file,
    )

    # Load extractor
    extractor = wneSortingProject.get_kilosort_extractor(
        wneSubject.name,
        experiment,
        alias,
        probe,
        sorting,
        postprocessing=postprocessing,
    )

    if wneHypnogramProject is not None:
        hypnogram = wneHypnogramProject.load_hypnogram(
            experiment, wneSubject.name, simplify=True
        )._df
    else:
        hypnogram = None

    # TODO Keep???
    # Fix extractor
    # extractor = units.si_ks_sorting.fix_isi_violations_ratio(extractor)
    # extractor = units.si_ks_sorting.fix_noise_cluster_labels(extractor)
    # extractor = units.si_ks_sorting.fix_uncurated_cluster_labels(extractor)

    # Add anatomy to the extractor, if available.
    if wneAnatomyProject is not None:
        anatomy_file = wneAnatomyProject.get_experiment_subject_file(
            experiment, wneSubject.name, f"{probe}.structures.htsv"
        )
        assert anatomy_file.exists(), (
            f"Could not find anatomy file at: {anatomy_file}.\n"
            f"Set `wneAnatomyProject = None` in kwargs to ignore anatomy."
        )
        structs = utils.read_htsv(anatomy_file)
    else:
        structs = None

    return units.SpikeInterfaceKilosortSorting(
        extractor, sample2time, hypnogram=hypnogram, structs=structs
    )


def load_multiprobe_sorting(
    wneSortingProject: Project,
    wneSubject: sglx.Subject,
    experiment: str,
    alias: str,
    probes: list[str],
    sortings: dict[str, str] = None,
    postprocessings: dict[str, str] = None,
    wneAnatomyProject: Optional[Project] = None,
    wneHypnogramProject: Optional[Project] = None,
    allow_no_sync_file=True,
) -> units.MultiprobeSorting:

    if sortings is None:
        sortings = {prb: None for prb in probes}
    if postprocessings is None:
        postprocessings = {prb: None for prb in probes}

    return units.MultiprobeSorting(
        {
            probe: load_singleprobe_sorting(
                wneSortingProject,
                wneSubject,
                experiment,
                alias,
                probe=probe,
                sorting=sortings[probe],
                postprocessing=postprocessings[probe],
                wneAnatomyProject=wneAnatomyProject,
                wneHypnogramProject=wneHypnogramProject,
                allow_no_sync_file=allow_no_sync_file,
            )
            for probe in probes
        }
    )
