from pathlib import Path

import pandas as pd
import spikeinterface.extractors as se

import ecephys.utils
import spikeinterface as si
from ecephys import wne
from ecephys.wne.sglx.project import SGLXProject
from ecephys.wne.sglx.subject import SGLXSubject

# These are saved as annotations on the SpikeGLX recording object
# There is one of these dicts per "slice" that was used to build the recording.
RECORDING_ANNOTATION_COLS = {
    # These allow one to recompute the timestamps
    "imSampRate": float,  # Slice original sampling rate.
    "n_slice_samples": int,  # Number of samples in the slice.
    "time_offset": float,  # Unsync'd start time of the slice (probe's acquisition clock).
    "sync_slope": float,  # Sync slope to convert from acquisition clock to master clock.
    "sync_intercept": float,  # Sync intercept to convert from acquisition clock to master clock.
    # These columns provide potentially useful provenance and validation information.
    "withinFileStartFrame": int,  # Start frame of the slice within the original file.
    "withinFileEndFrame": int,  # End frame of the slice within the original file.
    "expmtPrbAcqFirstTime": float,  # Unsync'd start time of the slice's file (probe's acquisition clock).
    "neo_segment_index": int,  # Neo segment index of the slice's file.
    "fileSHA1": str,  # SHA1 hash of the slice's file.
    "session": str,  # Session identifier.
    "run": str,  # Run identifier.
    "gate": str,  # Gate identifier.
    "trigger": str,  # Trigger identifier.
    "path": str,  # File path.
}


def get_recording(
    project: SGLXProject,
    subject: SGLXSubject,
    experiment: str,
    probe: str,
    stream: str = "ap",
) -> tuple[si.ConcatenateSegmentRecording, pd.DataFrame]:
    """
    Create a SpikeInterface recording object from a possibly-discontinuous multi-file
    recoring, with artifacts dropped, and timestamps that accurately reflect all
    excisions and synchronization to the canonical timebase.

    Parameters
    ----------
    project : SGLXProject
        The SGLX project instance.
    subject : SGLXSubject
        The SGLX subject instance.
    experiment : str
        The experiment identifier.
    probe : str
        The probe identifier.

    Returns
    -------
    si.ConcatenateSegmentRecording
        The SpikeInterface recording object.
    slices : pd.DataFrame
        The slice table used to build the recording. Only includes retained slices.
    """
    # Load artifacts, which will be dropped from the recording
    artifacts_file = project.get_experiment_subject_file(
        experiment,
        subject.name,
        f"{probe}.{stream}.{wne.Files.ARTIFACTS}",
    )
    artifacts = wne.utils.get_dummy_artifacts_table()
    if artifacts_file.exists():
        _artifacts = ecephys.utils.read_htsv(artifacts_file)
        artifacts = pd.concat([artifacts, _artifacts], ignore_index=True)

    # Figure out how each file needs to be sliced to drop artifacts
    ftab = subject.get_experiment_frame(
        experiment, alias="full", stream=stream, ftype="bin", probe=probe
    )
    slices = wne.sglx.utils.create_slice_table_for_spikeinterface(ftab, artifacts)

    # Determine neo segment index for each slice. One neo segment may have >1 slice.
    stream_id = f"{probe}.{stream}"
    for prb_dir in slices["gate_dir"].unique():
        gate_dir = prb_dir.parent  # The actual gate directory
        extractor = se.SpikeGLXRecordingExtractor(gate_dir, stream_id=stream_id)
        nseg = extractor.get_num_segments()
        for seg_idx in range(nseg):
            seg_info = extractor.neo_reader.signals_info_dict[(seg_idx, stream_id)]
            seg_bin_file = Path(
                seg_info["bin_file"]
            ).name  # Do not to use seg_info["fname"] nor seg_info["meta"]["fileName"],
            # as the former is parsed from the latter, and neither will match ftab if
            # the file has been renamed or symlinked with a new name.
            slices.loc[slices["fname"] == seg_bin_file, "neo_segment_index"] = seg_idx
    slices["neo_segment_index"] = slices["neo_segment_index"].astype(int)

    # "neo_segment_index" is equivalent to the old "gate_dir_trigger_file_idx":
    # assert all(fslices["gate_dir_trigger_file_idx"] == fslices["neo_segment_index"])

    # For each slice to keep, create a FrameSliceRecording, then concatenate them
    recordings = list()
    for fslice in slices.itertuples():
        probe_dir = fslice.gate_dir  # Actually the probe directory
        gate_dir = probe_dir.parent  # The actual gate directory
        extractor = se.SpikeGLXRecordingExtractor(gate_dir, stream_id=stream_id)
        segment = extractor.select_segments([fslice.neo_segment_index])
        recording = segment.frame_slice(
            start_frame=fslice.withinFileStartFrame,
            end_frame=fslice.withinFileEndFrame,
        )
        recordings.append(recording)
    recording = si.ConcatenateSegmentRecording(
        recordings, sampling_frequency_max_diff=1e-6
    )

    # Compute synchronzied timestamps for the concatenated recording.
    sync_file = project.get_experiment_subject_file(
        experiment, subject.name, wne.constants.Files.AP_SYNC
    )
    sync_table = ecephys.utils.read_htsv(sync_file)
    slices = wne.sglx.utils.add_sample2time_columns(slices, sync_table)
    times = wne.sglx.utils.slice_table2times(slices)
    recording.set_times(times, with_warning=False)

    # Annotate the recording with provenance information for each slice.
    annotations = slices[RECORDING_ANNOTATION_COLS.keys()].astype(
        RECORDING_ANNOTATION_COLS
    )
    recording.annotate(spikeglx_provenance=annotations.to_json())

    return recording, slices
