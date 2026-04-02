from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    import spikeinterface as si
    from ecephys.wne.sglx.project import SGLXProject
    from ecephys.wne.sglx.subject import SGLXSubject


def _apply_neo_spikeglx_filename_patch():
    """Monkey-patch neo's ``extract_stream_info`` to handle SpikeGLX metadata
    files whose ``fileName`` field contains spaces.

    Problem
    -------
    Neo 0.14.4's ``SpikeGLXRawIO._parse_header()`` calls ``scan_files()``
    which calls ``extract_stream_info()`` for every ``.meta`` file it finds.
    ``extract_stream_info`` reads the ``fileName`` key from the metadata to
    obtain the original recording path, then passes its stem to
    ``parse_spikeglx_fname()`` to extract ``gate_num`` and ``trigger_num``.

    For CNPIX7-Giuseppe, the metadata ``fileName`` still contains the
    *original* Windows recording path, which has spaces and dots::

        fileName=F:/CNPIX7/12.12.2020 BL 24hs_g0/.../12.12.2020 BL 24hs_g0_t0.imec0.ap.bin

    The files on disk were renamed to use dashes and underscores::

        12-12-2020_BL_24hs_g0_t0.imec0.ap.bin

    All of neo's ``parse_spikeglx_fname`` regex patterns use ``\\S+`` (one or
    more non-whitespace characters) for the run-name token. Because the
    original filename has spaces, none of the standard patterns match, and the
    parser falls through to a generic fallback that returns
    ``gate_num=None, trigger_num=None``.

    ``_add_segment_order`` then maps *every* ``(None, None)`` tuple to
    ``seg_index=0``, so all trigger files in the gate directory collide on
    the same ``(seg_index, stream_name)`` key, raising::

        KeyError: "key (0, 'imec0.ap') is already in the signals_info_dict"

    Fix
    ---
    This patch wraps ``extract_stream_info`` so that when the metadata
    ``fileName`` fails to produce valid ``gate_num``/``trigger_num`` values,
    it retries parsing using the *actual* on-disk filename (``meta_file``),
    which has been renamed to a well-formed SpikeGLX name that neo can parse.

    Neo is installed from PyPI (not a local fork), so a monkey-patch is
    necessary. The patch is idempotent (safe to call multiple times) and
    only activates for files where the metadata ``fileName`` cannot be parsed
    — subjects with well-formed metadata are completely unaffected.
    """
    import neo.rawio.spikeglxrawio as sglx_rawio

    original_fn = sglx_rawio.extract_stream_info
    if getattr(original_fn, "_patched_for_filename_spaces", False):
        return  # Already applied.

    def _patched_extract_stream_info(meta_file, meta):
        info = original_fn(meta_file, meta)
        if info["gate_num"] is None or info["trigger_num"] is None:
            # The metadata fileName couldn't be parsed. Try the actual
            # on-disk filename, which may have been renamed to a parseable
            # SpikeGLX name.
            disk_fname = Path(meta_file).stem  # e.g. "..._g0_t0.imec0.ap"
            try:
                _, gate_num, trigger_num, _, _ = sglx_rawio.parse_spikeglx_fname(
                    disk_fname
                )
            except ValueError:
                return info  # On-disk name also unparseable; nothing we can do.
            if gate_num is not None:
                info["gate_num"] = gate_num
            if trigger_num is not None:
                info["trigger_num"] = trigger_num
        return info

    _patched_extract_stream_info._patched_for_filename_spaces = True
    sglx_rawio.extract_stream_info = _patched_extract_stream_info


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
) -> tuple["si.ConcatenateSegmentRecording", pd.DataFrame]:
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
    _apply_neo_spikeglx_filename_patch()

    from spikeinterface.extractors.extractor_classes import SpikeGLXRecordingExtractor

    import spikeinterface as si
    from ecephys import utils as ece_utils
    from ecephys.wne import Files as wne_files
    from ecephys.wne import utils as wne_utils
    from ecephys.wne.sglx import utils as sglx_utils

    # Load artifacts, which will be dropped from the recording
    artifacts_file = project.get_experiment_subject_file(
        experiment,
        subject.name,
        f"{probe}.{stream}.{wne_files.ARTIFACTS}",
    )
    artifacts = wne_utils.get_dummy_artifacts_table()
    if artifacts_file.exists():
        _artifacts = ece_utils.read_htsv(artifacts_file)
        artifacts = pd.concat([artifacts, _artifacts], ignore_index=True)

    # Figure out how each file needs to be sliced to drop artifacts
    ftab = subject.get_experiment_frame(
        experiment, alias="full", stream=stream, ftype="bin", probe=probe
    )
    slices = sglx_utils.create_slice_table_for_spikeinterface(ftab, artifacts)

    # Determine neo segment index for each slice. One neo segment may have >1 slice.
    stream_id = f"{probe}.{stream}"
    for prb_dir in slices["gate_dir"].unique():
        gate_dir = prb_dir.parent  # The actual gate directory
        extractor = SpikeGLXRecordingExtractor(gate_dir, stream_id=stream_id)
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
        extractor = SpikeGLXRecordingExtractor(gate_dir, stream_id=stream_id)
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
        experiment, subject.name, wne_files.AP_SYNC
    )
    sync_table = ece_utils.read_htsv(sync_file)
    slices = sglx_utils.add_sample2time_columns(slices, sync_table)
    times = sglx_utils.slice_table2times(slices)
    recording.set_times(times, with_warning=False)

    # Annotate the recording with provenance information for each slice.
    annotations = slices[RECORDING_ANNOTATION_COLS.keys()].astype(
        RECORDING_ANNOTATION_COLS
    )
    recording.annotate(spikeglx_provenance=annotations.to_json())

    return recording, slices
