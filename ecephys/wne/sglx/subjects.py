import gzip
import pickle
import itertools as it
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

import spikeinterface as si
import spikeinterface.extractors as se
from ecephys import utils
from ecephys.sglx import file_mgmt
from ecephys.wne.sglx import experiments, sessions

logger = logging.getLogger(__name__)


class SGLXSubject:
    """The cache contains the sessions frame."""

    def __init__(
        self, subjectYamlFile: Path, subjectCache: Optional[pd.DataFrame] = None
    ):
        self.name = subjectYamlFile.stem
        self.doc = SGLXSubject.load_yaml_doc(subjectYamlFile)
        self.cache = self.refresh_cache() if subjectCache is None else subjectCache

    def __repr__(self) -> str:
        return f"SGLXSubject: {self.name}"

    def refresh_cache(self) -> pd.DataFrame:
        logger.debug(f"Refreshing cache for: {self.name}")
        sessionFrames = {
            sessionDict["id"]: file_mgmt.filelist_to_frame(
                sessions.get_session_files_from_multiple_locations(sessionDict)
            ).assign(imSyncType=sessionDict.get("imSyncType", None))
            for sessionDict in self.doc["recording_sessions"]
        }
        self.cache = (
            pd.concat(sessionFrames, names=["session"])
            .reset_index(level=0)
            .reset_index(drop=True)
        )
        return self.cache

    def get_session_frame(self, session_id: str, **kwargs) -> pd.DataFrame:
        frame = self.cache.loc[self.cache["session"] == session_id]
        return file_mgmt.loc(frame, **kwargs).reset_index(drop=True)

    def get_experiment_names(self) -> list[str]:
        return list(self.doc["experiments"].keys())

    def get_experiment_probes(self, experimentName) -> list[str]:
        return list(self.get_experiment_frame(experimentName)["probe"].unique())

    def get_experiment_session_ids(self, experimentName) -> list[str]:
        return self.doc["experiments"][experimentName]["recording_session_ids"]

    def get_experiment_frame(
        self,
        experiment: str,
        alias: Optional[str] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Get all SpikeGLX files matching selection criteria."""
        sessionIDs = self.get_experiment_session_ids(experiment)
        frame = self.cache[
            self.cache["session"].isin(sessionIDs)
        ]  # Get the cache slice containing this experiment.
        if frame.empty:
            logger.info(
                f"No frame found for {self.name}: {experiment} with recording session IDs: {sessionIDs} \n"
                "There is probably a problem with this subject's YAML file."
            )
        frame = experiments.add_experiment_times(frame)
        # This exists to get around limitations of SpikeInterface, so can hopefully be removed one day.
        frame = _get_gate_dir_trigger_file_index(frame)

        if alias is not None:
            subaliases = self.doc["experiments"][experiment]["aliases"][alias]
            if not isinstance(subaliases, list):
                raise ValueError(
                    f"Alias {alias} must be specified as a list of subaliases, even if there is only a single subalias."
                )
            subaliasFrames = [
                sessions.get_subalias_frame(frame, sa) for sa in subaliases
            ]
            frame = pd.concat(subaliasFrames).reset_index(drop=True)

        return file_mgmt.loc(frame, **kwargs).reset_index(drop=True)

    def get_lfp_bin_paths(self, experiment: str, alias=None, **kwargs) -> list[Path]:
        return self.get_experiment_frame(
            experiment, alias, stream="lf", ftype="bin", **kwargs
        ).path.values

    def get_ap_bin_paths(self, experiment: str, alias=None, **kwargs) -> list[Path]:
        return self.get_experiment_frame(
            experiment, alias, stream="ap", ftype="bin", **kwargs
        ).path.values

    def get_lfp_bin_table(self, experiment: str, alias=None, **kwargs) -> pd.DataFrame:
        return self.get_experiment_frame(
            experiment, alias, stream="lf", ftype="bin", **kwargs
        )

    def get_ap_bin_table(self, experiment: str, alias=None, **kwargs) -> pd.DataFrame:
        return self.get_experiment_frame(
            experiment, alias, stream="ap", ftype="bin", **kwargs
        )

    def get_experiment_data_times(
        self, experiment: str, probe: str, as_datetimes=False
    ) -> tuple:
        df = self.get_experiment_frame(experiment, probe=probe)
        if as_datetimes:
            return (
                df["expmtPrbAcqFirstDatetime"].min(),
                df["expmtPrbAcqLastDatetime"].max(),
            )
        else:
            return (df["expmtPrbAcqFirstTime"].min(), df["expmtPrbAcqLastTime"].max())

    def t2dt(self, experiment: str, probe: str, t):
        dt0, _ = self.get_experiment_data_times(experiment, probe, as_datetimes=True)
        return pd.to_timedelta(t, "s") + dt0

    def dt2t(self, experiment: str, probe: str, dt):
        dt0, _ = self.get_experiment_data_times(experiment, probe, as_datetimes=True)
        return (dt - np.datetime64(dt0)) / pd.to_timedelta("1s")

    def get_alias_datetimes(self, experiment: str, alias: str) -> list[tuple]:
        subaliases = self.doc["experiments"][experiment]["aliases"][alias]

        def get_subalias_datetimes(subalias: dict) -> tuple:
            if not ("start_time" in subalias) and ("end_time" in subalias):
                raise NotImplementedError(
                    f"All subaliases of {alias} must have start_time and end_time fields."
                )
            return (
                pd.to_datetime(subalias["start_time"]),
                pd.to_datetime(subalias["end_time"]),
            )

        return [get_subalias_datetimes(sa) for sa in subaliases]

    def get_si_recording(
        self,
        experiment: str,
        alias: str,
        stream: str,
        probe: str,
        combine: str = "concatenate",
        exclusions: pd.DataFrame = pd.DataFrame(
            {"fname": [], "withinFileStartTime": [], "withinFileEndTime": [], "type": []}
        ),
        sampling_frequency_max_diff: Optional[float] = 0,  # TODO: Should this be 1e-6?
    ) -> tuple[si.BaseRecording, pd.DataFrame]:
        """Combine the one or more recordings comprising an experiment or alias into a single SI recording object.

        Parameters
        ==========
        combine: 'concatenate' or 'append'
            If 'concatenate' (default), the returned recording object is one single monolothic segment.
            This is the default behavior, because SI sorters currently only work on single-segment recordings.
            If 'append', the returned recording object consists of multiple segments.
            This might be useful for certain preprocessing, postprocessing operations, etc.
        exclusions:
            Specify which parts of the recording to drop. We slice such that the first and last samples
            of each exclusion are NOT included in the returned recording.
            fname: The name of the file (e.g. 3-2-2021_J_g0_t1.imec1.ap.bin)
            withinFileStartTime: The start time of the data to drop, in seconds from the start of the file.
            withinFileEndTime: The end time of the data to drop, in seconds from the start of the file.
                If greater than the file duration, the excess time will be ignored, not dropped from the next file.
            type: A label you can assign to keep track of why this data was excluded. As long as the value is not "keep", the data will be dropped.

        Returns
        =======
        recording:
            The combined SI recording object.
        segments:
            A dataframe where each row is a segment of data to keep, or drop, sorted in chronological order.
                fname: The name of the file (e.g. 3-2-2021_J_g0_t1.imec1.ap.bin)
                withinFileStartFrame: The first sample index of the segment, measured from the start of the file (0-indexed)
                withinFileEndFrame: The final sample index of the segment, measured from the start of the file (0-indexed)
                type: Either 'keep', in which case the segment was kept, or other, in which case the segment was dropped.
                segmentDuration: Duration in sec of segment.
        """
        # Get the experiment frame. This should be for a single probe, and a single stream.
        ftab = self.get_experiment_frame(
            experiment, alias=alias, stream=stream, ftype="bin", probe=probe
        )
        # Split the experiment frame around the exclusions, using precise sample indices.
        segments = segment_experiment_frame_for_spikeinterface(ftab, exclusions)

        # Take the good segments one by one, create an recording object for each, and save these all in a list
        good_segments = segments[segments["type"] == "keep"]
        recordings = list()
        for segment in good_segments.itertuples():
            extractor = se.SpikeGLXRecordingExtractor(
                segment.gate_dir, stream_id=f"{probe}.{stream}"
            )
            recording = extractor.select_segments(
                [segment.gate_dir_trigger_file_idx]
            ).frame_slice(start_frame=segment.withinFileStartFrame, end_frame=segment.withinFileEndFrame)
            recordings.append(recording)

        # Combine the good segments
        if combine == "concatenate":
            fn = si.concatenate_recordings
        elif combine == "append":
            fn = si.append_recordings
        else:
            raise ValueError(f"Got unexpected value for `combine`: {combine}")

        recording = fn(
            recordings, sampling_frequency_max_diff=sampling_frequency_max_diff
        )

        # We return both recording and segments together, rather than making the available separately,
        # to ensure that you never get a segment table unless it is actually proven to produce a valid extractor object.
        return recording, segments

    def get_tdt_block_path(self, experiment: str):
        return Path(self.doc["experiments"][experiment]["tdt_block_path"])

    @staticmethod
    def load_yaml_doc(yaml_path: Path) -> dict:
        """Load a YAML file that contains only one document."""
        with open(yaml_path) as fp:
            yaml_doc = yaml.safe_load(fp)
        return yaml_doc


class SGLXSubjectLibrary:
    def __init__(self, libdir: Path):
        self.libdir = libdir
        self.cachefile = self.libdir / "wne_sglx_cache.gz"
        self.cache = (
            self.read_cache() if self.cachefile.is_file() else None
        )

    def get_subject_file(self, subjectName: str) -> Path:
        return self.libdir / f"{subjectName}.yml"

    def get_subject(self, subjectName: str) -> SGLXSubject:
        subjectFrame = (
            self.cache[self.cache["subject"] == subjectName]
            .drop(columns="subject")
            .reset_index(drop=True)
            if self.cache is not None
            else None
        )
        return SGLXSubject(self.get_subject_file(subjectName), subjectFrame)

    def get_subject_names(self) -> list[str]:
        return sorted([f.stem for f in self.libdir.glob("*.yml")])

    def refresh_cache(self) -> pd.DataFrame:
        names = self.get_subject_names()
        subjectCaches = []
        for name in names:
            logger.debug(f"Refreshing cache for {name}")
            subjectCaches.append(SGLXSubject(self.get_subject_file(name)).cache)
        self.cache = pd.concat(
            subjectCaches, keys=names, names=["subject"]
        ).reset_index(level=0)
        return self.cache

    # TODO: This would be better written as an HTSV or PQT, with datatypes assigned during loading.
    def write_cache(self):
        if self.cache is None:
            self.refresh_cache()
        # Compress because we overpassed github's filesize limit
        with gzip.open(self.cachefile, "wb") as f:
            pickle.dump(self.cache, f)
    
    def read_cache(self):
        # Compress because we overpassed github's filesize limit
        with gzip.open(self.cachefile, "rb") as f:
            return pickle.load(f)


# TODO: Remove as soon as SpikeInterface adds this functionality.
# 1/30/2023 Tom says It is still necessary, because although the relevant GitHub issues seem to have been closed, the API has not changed.
def _get_gate_dir_trigger_file_index(ftab: pd.DataFrame) -> pd.DataFrame:
    """Get index of trigger file relative to all files of same stream/prb/gate_folder.

    This is relative to files currently present in the directory, so we can't
    just parse trigger index (doesn't work if some files are moved).

    Useful to instantiate spikeinterface extractor objects, since they require
    subselecting segments of interest (ie trigger files) after instantiation.
    See https://github.com/SpikeInterface/spikeinterface/issues/628#issuecomment-1130232542

    """
    ftab["gate_dir"] = ftab.apply(lambda row: row["path"].parent, axis=1)
    for gate_dir, prb, stream, ftype in it.product(
        ftab.gate_dir.unique(),
        ftab.probe.unique(),
        ftab.stream.unique(),
        ftab.ftype.unique(),
    ):
        mask = (
            (ftab["gate_dir"] == gate_dir)
            & (ftab["probe"] == prb)
            & (ftab["stream"] == stream)
            & (ftab["ftype"] == ftype)
        )
        mask_n_triggers = int(mask.sum())
        ftab.loc[mask, "gate_dir_n_trigger_files"] = mask_n_triggers
        ftab.loc[mask, "gate_dir_trigger_file_idx"] = np.arange(0, mask_n_triggers)

    ftab["gate_dir_n_trigger_files"] = ftab["gate_dir_n_trigger_files"].astype(int)
    ftab["gate_dir_trigger_file_idx"] = ftab["gate_dir_trigger_file_idx"].astype(int)

    return ftab


def segment_experiment_frame_for_spikeinterface(
    ftab: pd.DataFrame, exclusions: pd.DataFrame
) -> pd.DataFrame:
    """Split an experiment frame for a single probe, steam, and filetype around a set of periods to exclude.
    For details, see `get_si_recording()`.
    """
    EXCLUSION_COLS = ["withinFileStartTime", "withinFileEndTime", "fname"]
    assert all([c in exclusions.columns for c in EXCLUSION_COLS]), (
        f"Invalid columns for exclusions. Expected: `{EXCLUSION_COLS}`"
    )
    segments = list()
    # For each file in the experiment, split it if necessary.
    # If not, just create a segment that is the entire file.
    for file in ftab.itertuples():
        ns = file.nFileSamp
        fname = file.path.name
        mask = (
            exclusions["fname"] == fname
        )  # Get the exclusions pertaining to this file.

        # For the exclusions pertaining to this file, convert their definition in seconds to precise sample indices,
        # and clip these estimates so that sample indices don't extend beyond the ends of the file.
        exclusions.loc[mask, "withinFileStartFrame"] = (
            (exclusions.loc[mask, "withinFileStartTime"] * file.imSampRate)
            .astype(int)
            .clip(0, ns)
        )
        exclusions.loc[mask, "withinFileEndFrame"] = (
            (exclusions.loc[mask, "withinFileEndTime"] * file.imSampRate).astype(int).clip(0, ns)
        )

        # Do the actual splitting of the entire file around the exclusions
        file_segments = utils.reconcile_labeled_intervals(
            exclusions.loc[mask, ["withinFileStartFrame", "withinFileEndFrame", "type"]],
            pd.DataFrame({"withinFileStartFrame": [0], "withinFileEndFrame": [ns], "type": "keep"}),
            "withinFileStartFrame",
            "withinFileEndFrame",
        ).drop(columns="delta")
        file_segments["fname"] = fname

        # The function above considers intervals to be open-ended, so that (a, b) and (b, c) are considered NON-overlapping
        # (as for usual python slicing)
        # This means that up to this point the end sample of an exclusion will be part of the next (kept) segment.
        # In order to be conservative, we correct each bad segment followed by a good segment to include its last sample.
        # For example, if (a, b) and (b, c) are bad segment, and (c, d) is a good segment, the new segments will be (a, b+1), (b+1, c), (c+1, d).
        keep = file_segments["type"] == "keep"
        frames_to_shift = np.intersect1d(
            file_segments[~keep]["withinFileEndFrame"].values,
            file_segments["withinFileStartFrame"].values,
        )  # End of each bad segment followed by another segment (excludes the last one)
        i = file_segments["withinFileEndFrame"].isin(frames_to_shift)
        j = file_segments["withinFileStartFrame"].isin(frames_to_shift)
        file_segments.loc[i, "withinFileEndFrame"] += 1
        file_segments.loc[j, "withinFileStartFrame"] += 1

        # Do some sanity checks, ensuring that every sample in the file is accounted for.
        assert (
            file_segments["withinFileStartFrame"].min() == 0
        ), "Something went wrong when splitting file around exclusions."
        assert file_segments["withinFileEndFrame"].max() == (
            ns
        ), "Something went wrong when splitting file around exclusions."
        segments.append(file_segments)
        assert (
            file_segments["withinFileEndFrame"] - file_segments["withinFileStartFrame"]
        ).sum() == ns, "Something went wrong when splitting file around exclusions."

    # Return the segments, adding metadata about the files that they come from, for convenience.
    segments = pd.concat(segments, ignore_index=True).astype(
        {"withinFileStartFrame": int, "withinFileEndFrame": int}
    )
    ftab["fname"] = ftab["path"].apply(lambda x: x.name)

    stab = segments.merge(ftab, on="fname")

    stab["nSegmentSamp"] = stab["withinFileEndFrame"] - stab["withinFileStartFrame"]
    stab["segmentDuration"] = stab["nSegmentSamp"].div(stab["imSampRate"])

    return stab
