import logging
import pathlib
from typing import Callable

import numpy as np

from ecephys import utils
from ecephys.wne import Project

logger = logging.getLogger(__name__)


class SGLXProject(Project):
    def __init__(self, project_name: str, project_dir: pathlib.Path):
        Project.__init__(self, project_name, project_dir)

    def load_segments_table(
        self,
        subject: str,
        experiment: str,
        alias: str,
        probe: str,
        sorting: str,
        return_all_segment_types=False,
    ):
        """Load a sorting's segment file.

        Add a couple useful columns: `nSegmentSamp`, `segmentDuration`, `segmentExpmtPrbAcqFirstTime`, `segmentExpmtPrbAcqLastTime`
        """
        segment_file = (
            self.get_alias_subject_directory(experiment, alias, subject)
            / f"{sorting}.{probe}"
            / "segments.htsv"
        )
        if not segment_file.exists():
            raise FileNotFoundError(f"Segment table not found at {segment_file}.")

        segments = utils.read_htsv(segment_file)

        segments["nSegmentSamp"] = segments["end_frame"] - segments["start_frame"]
        segments["segmentDuration"] = segments["nSegmentSamp"].div(
            segments["imSampRate"]
        )
        segments["segmentExpmtPrbAcqFirstTime"] = segments[
            "expmtPrbAcqFirstTime"
        ] + segments["start_frame"].div(segments["imSampRate"])
        segments["segmentExpmtPrbAcqLastTime"] = (
            segments["segmentExpmtPrbAcqFirstTime"] + segments["segmentDuration"]
        )

        if return_all_segment_types:
            return segments

        return segments[segments["type"] == "keep"]

    def get_sample2time(
        self,
        subject: str,
        experiment: str,
        alias: str,
        probe: str,
        sorting: str,
        allow_no_sync_file=False,
    ) -> Callable:
        # TODO: Once permissions are fixed, should come from f = wneProject.get_experiment_subject_file(experiment, subject, f"prb_sync.ap.htsv")
        # Load probe sync table.
        probe_sync_file = (
            self.get_alias_subject_directory(experiment, alias, subject)
            / f"{sorting}.{probe}"
            / "prb_sync.ap.htsv"
        )
        if not probe_sync_file.exists():
            if allow_no_sync_file:
                logger.info(
                    f"Could not find sync table at {probe_sync_file}.\n"
                    f"`allow_no_sync_file` == True : Ignoring probe sync in sample2time"
                )
                sync_table = None
            else:
                logger.info(f"Could not find sync table at {probe_sync_file}.")
                return None
        else:
            sync_table = utils.read_htsv(
                probe_sync_file
            )  # Used to map this probe's times to imec0.

        # Load segment table
        segments = self.load_segments_table(
            subject,
            experiment,
            alias,
            probe,
            sorting,
            return_all_segment_types=False,
        )  # Used to map SI sorting samples to this probe's times.

        # Get all the good segments (aka the ones in the sorting), in chronological order.
        # Compute which samples in the recording belong to each segment.
        sorted_segments = segments[segments["type"] == "keep"].copy()
        sorted_segments["nSegmentSamples"] = (
            sorted_segments["end_frame"] - sorted_segments["start_frame"]
        )  # N of sorted samples in each segment

        cum_sorted_samples_by_end = sorted_segments[
            "nSegmentSamples"
        ].cumsum()  # N of sorted samples by the end of each segment
        cum_sorted_samples_by_start = cum_sorted_samples_by_end.shift(
            1, fill_value=0
        )  # N of sorted samples by the start of each segment
        sorted_segments[
            "start_sample"
        ] = cum_sorted_samples_by_start  # First sample index of concatenated recording belonging to each semgent
        sorted_segments["end_sample"] = cum_sorted_samples_by_end

        # Given a sample number in the SI recording, we can now figure out:
        #   (1) the segment it came from
        #   (2) the file that segment belongs to
        #   (3) how to map that file's times into our canonical timebase.
        # We make a function that does this for an arbitrary array of sample numbers in the SI object, so we can use it later as needed.
        if sync_table is not None:
            sync_table = sync_table.set_index("source")

        # TODO: Rename start_sample -> si_start_sample?
        def sample2time(s):
            s = s.astype("float")
            t = np.empty(s.size, dtype="float")
            t[:] = np.nan  # Check a posteriori if we covered all input samples
            for seg in sorted_segments.itertuples():
                mask = (s >= seg.start_sample) & (
                    s < seg.end_sample
                )  # Mask samples belonging to this segment
                t[mask] = (
                    (s[mask] - seg.start_sample) / seg.imSampRate
                    + seg.expmtPrbAcqFirstTime
                    + seg.start_frame / seg.imSampRate
                )  # Convert to number of seconds in this probe's (expmtPrbAcq) timebase
                if sync_table is not None:
                    sync_entry = sync_table.loc[
                        seg.fname
                    ]  # Get info needed to sync to imec0's (expmtPrbAcq) timebase
                    t[mask] = (
                        sync_entry.slope * t[mask] + sync_entry.intercept
                    )  # Sync to imec0 (expmtPrbAcq) timebase
            assert not any(np.isnan(t)), (
                "Some of the provided sample indices were not covered by segments \n"
                "and therefore couldn't be converted to time"
            )

            return t

        return sample2time
