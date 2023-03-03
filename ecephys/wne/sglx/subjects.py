import logging
from pathlib import Path
from typing import Optional

import numpy as np
import numpy.testing as npt
import pandas as pd
import yaml

import ecephys as ece
import spikeinterface as si
from ecephys.utils.spikeinterface_utils import \
    load_single_segment_sglx_recording

logger = logging.getLogger(__name__)


class Subject:
    def __init__(self, subjectYamlFile: Path, subjectCache=None):
        self.name = subjectYamlFile.stem
        self.doc = Subject.load_yaml_doc(subjectYamlFile)
        self.cache = self.refresh_cache() if subjectCache is None else subjectCache

    def __repr__(self) -> str:
        return f"wneSubject: {self.name}"

    def refresh_cache(self) -> pd.DataFrame:
        sessionFrames = [
            ece.sglx.filelist_to_frame(
                ece.wne.sglx.sessions.get_session_files_from_multiple_locations(
                    sessionDict
                )
            )
            for sessionDict in self.doc["recording_sessions"]
        ]
        self.cache = pd.concat(
            sessionFrames,
            keys=[session["id"] for session in self.doc["recording_sessions"]],
            names=["session"],
        ).reset_index(level=0)
        return self.cache

    def get_experiment_names(self) -> list[str]:
        return list(self.doc["experiments"].keys())

    def get_experiment_probes(self, experimentName) -> list[str]:
        return list(self.get_file_frame(experimentName)["probe"].unique())

    def get_file_frame(
        self,
        experimentName: str,
        aliasName=None,
        **kwargs,
    ) -> pd.DataFrame:
        """Get all SpikeGLX files matching selection criteria.

        To check that all returned files are continguous (to within the default tolerance of `add_wne_times`, i.e. 1 sample):
        assert frame["isContinuation"][1:].all()
        """
        sessionIDs = self.doc["experiments"][experimentName]["recording_session_ids"]
        frame = self.cache[
            self.cache["session"].isin(sessionIDs)
        ]  # Get the cache slice corresponding to this experiment.
        frame = ece.wne.sglx.add_wne_times(frame)
        # TODO: The following is not a function of the experiment, so should probably be done elsewhere.
        # Also, this exists to get around limitations of SpikeInterface, so can hopefully be removed one day.
        frame = ece.wne.sglx.get_gate_dir_trigger_file_index(frame)

        if aliasName is not None:
            subaliases = self.doc["experiments"][experimentName]["aliases"][aliasName]
            if not isinstance(subaliases, list):
                raise ValueError(
                    f"Alias {aliasName} must be specified as a list of subaliases, even if there is only a single subalias."
                )
            subaliasFrames = [
                ece.wne.sglx.experiments.get_subalias_frame(frame, sa)
                for sa in subaliases
            ]
            frame = pd.concat(subaliasFrames).reset_index(drop=True)

        return ece.sglx.loc(frame, **kwargs).reset_index(drop=True)

    def get_lfp_bin_paths(self, experiment: str, alias=None, **kwargs) -> list[Path]:
        return self.get_file_frame(
            experiment, alias, stream="lf", ftype="bin", **kwargs
        ).path.values

    def get_ap_bin_paths(self, experiment: str, alias=None, **kwargs) -> list[Path]:
        return self.get_file_frame(
            experiment, alias, stream="ap", ftype="bin", **kwargs
        ).path.values

    def get_lfp_bin_table(self, experiment: str, alias=None, **kwargs) -> pd.DataFrame:
        return self.get_file_frame(
            experiment, alias, stream="lf", ftype="bin", **kwargs
        )

    def get_ap_bin_table(self, experiment: str, alias=None, **kwargs) -> pd.DataFrame:
        return self.get_file_frame(
            experiment, alias, stream="ap", ftype="bin", **kwargs
        )

    def get_experiment_data_times(
        self, experiment: str, probe: str, as_datetimes=False
    ) -> tuple:
        df = self.get_file_frame(experiment, probe=probe)
        if as_datetimes:
            return (df["wneFileStartDatetime"].min(), df["wneFileEndDatetime"].max())
        else:
            return (df["wneFileStartTime"].min(), df["wneFileEndTime"].max())

    def t2dt(self, experiment: str, probe: str, t):
        dt0, _ = self.get_experiment_data_times(experiment, probe, as_datetimes=True)
        return pd.to_timedelta(t, "s") + dt0

    def dt2t(self, experiment: str, probe: str, dt):
        dt0, _ = self.get_experiment_data_times(experiment, probe, as_datetimes=True)
        return (dt - dt0) / pd.to_timedelta("1s")

    def get_alias_datetimes(self, experimentName: str, aliasName: str) -> list[tuple]:
        subaliases = self.doc["experiments"][experimentName]["aliases"][aliasName]

        def get_subalias_datetimes(subalias: dict) -> tuple:
            if not ("start_time" in subalias) and ("end_time" in subalias):
                raise NotImplementedError(
                    f"All subaliases of {aliasName} must have start_time and end_time fields."
                )
            return (
                pd.to_datetime(subalias["start_time"]),
                pd.to_datetime(subalias["end_time"]),
            )

        return [get_subalias_datetimes(sa) for sa in subaliases]


    def split_file_frame(
        self,
        fframe: pd.DataFrame,
        artifacts_frame: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Split file rows into segment around artifacts."""

        if artifacts_frame is None:
            artifacts_frame = pd.DataFrame()  # So we add columns of interest anyway
        else:
            assert all([c in artifacts_frame.columns for c in ["start", "stop", "file"]])

        segment_rows = []
        for _, frow in fframe.iterrows():

            # Artifacts of interest for each file
            fstem = f"{frow['run']}_{frow['gate']}_{frow['trigger']}"
            if len(artifacts_frame):
                fartifacts = artifacts_frame.loc[artifacts_frame["file"] == fstem]
            else:
                fartifacts = pd.DataFrame()

            # Iterate on file's artifacts
            srate = float(frow["imSampRate"])
            wneFileStartTime = float(frow["wneFileStartTime"])
            currentSegmentFileStartSample = int(0)
            currentSegmentFileStartTime = float(currentSegmentFileStartSample) / srate
            currentSegmentIdx = 0

            for _, arow in fartifacts.iterrows():
                
                aStart = float(arow["start"])
                assert aStart >= wneFileStartTime + currentSegmentFileStartTime
                assert aStart < float(frow["wneFileEndTime"])
                aStop = float(arow["stop"])
                if aStop >= float(frow["wneFileEndTime"]):
                    # Trim to end of file
                    aFileStopSamp = int(frow["nFileSamp"])
                else:
                    aFileStopSamp = int((aStop - wneFileStartTime) * srate)

                srow = frow.copy(deep=True)
                srow['segment_idx'] = currentSegmentIdx
                srow['segmentFileStartSample'] =  int(currentSegmentFileStartSample)
                srow['segmentFileEndSample'] =  int((aStart - wneFileStartTime) * srate)
                srow['nSegmentSamp'] =  srow['segmentFileEndSample'] - srow['segmentFileStartSample']

                # Prepare next: First sample after end of artifact
                currentSegmentIdx += 1
                currentSegmentFileStartSample = aFileStopSamp
                currentSegmentFileStartTime = currentSegmentFileStartSample / srate

                segment_rows.append(srow)

            # Last segment of file: end of artifact (or start of file) to end of file
            srow = frow.copy(deep=True)
            srow['segment_idx'] = currentSegmentIdx
            srow['segmentFileStartSample'] = int(currentSegmentFileStartSample)
            # Make sure we get to the last sample
            srow['segmentFileEndSample'] = int(frow["nFileSamp"]) # Up to end of file
            srow['nSegmentSamp'] =  int(srow["segmentFileEndSample"] - srow["segmentFileStartSample"])

            segment_rows.append(srow)

        sframe = pd.concat(segment_rows, axis=1).T

        # Convert Samples to FileTimes and wneTimes 
        sframe["segmentFileStartTime"] = sframe["segmentFileStartSample"].astype(float) / srate
        sframe["segmentFileEndTime"] = sframe["segmentFileEndSample"].astype(float) / srate
        sframe["wneSegmentStartTime"] = sframe["wneFileStartTime"].astype(float) + sframe["segmentFileStartTime"]
        sframe["wneSegmentEndTime"] = sframe["wneFileStartTime"].astype(float) + sframe["segmentFileEndTime"]
        # # Duration
        sframe["wneSegmentTimeSecs"] = sframe["wneSegmentEndTime"] - sframe["wneSegmentStartTime"]
        sframe["segmentTimeSecs"] = sframe["segmentFileEndTime"] - sframe["segmentFileStartTime"]
        # Ratio of total size
        sframe["segmentFileSizeRatio"] = sframe["nSegmentSamp"].astype(float) / sframe["nFileSamp"].astype(float)
        sframe["segmentIsFullFile"] = sframe["nSegmentSamp"].astype(int) == sframe["nFileSamp"].astype(int)

        # Remove empty segments
        mask = sframe["nSegmentSamp"] > 0
        sframe = sframe.loc[mask].reset_index()

        # Sanity check
        if artifacts_frame is None or not len(artifacts_frame):
            assert np.all(sframe["nSegmentSamp"] == sframe["nFileSamp"])
        assert sframe["wneSegmentStartTime"].is_monotonic_increasing

        return sframe


    def get_segment_frame(
        self,
        experimentName: str,
        aliasName: Optional[str] = None,
        artifacts_frame: Optional[pd.DataFrame] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Return frame split around artifacts.

        When there is no artifact in a file, the segment is the full file.
        The time of artifacts are approximate, as are the wne times, but we ensure
        that the first segment starts at the first file sample and the last segment ends
        at the end of file.

        The returned segment frame mimicks the file table with the following extra columns.
        - 'segment_idx': Index of segment in file.
        - 'segmentFileStartSample', 'segmentFileEndSample', 'nSegmentSamp': 
            Start, end, duration of segment relative to file start in samples (exact)
            Start is exclusive, end is exclusive as for usual python slices. 
        - 'segmentFileStartTime', 'segmentFileEndTime', 'segmentTimeSecs': 
            Start, end, duration of segment relative to file start (approximate)
        - 'wneSegmentStartTime', 'wneSegmentEndTime', 'wneSegmentTimeSecs':
            Start, end, duration time relative to experiment start (approximate)
        - 'segmentFileSizeRatio', 'segmentIsFullFile'
        """
        fframe = self.get_file_frame(
            experimentName,
        aliasName=aliasName,
            **kwargs,
        )
        return self.split_file_frame(
            fframe,
            artifacts_frame=artifacts_frame,
        )

    ### Spike interface recordings

    def _get_single_segment_si_recordings_list(
        self,
        experimentName : str,
        aliasName : str,
        stream : str,
        probe : str,
        time_ranges : Optional[list] = None,
        artifacts_frame : Optional[pd.DataFrame] = None,
    ):

        sframe = self.get_segment_frame(
            experimentName, 
            aliasName,
            artifacts_frame=artifacts_frame,
            probe=probe,
            stream=stream,
            ftype="bin",
        )

        if time_ranges is not None and artifacts_frame is not None:
            raise ValueError(
                "Can't provide both `time_ranges` and `artifacts_frame` kwargs"
            ) 
        if time_ranges is not None:
            assert len(time_ranges) == len(sframe)

        stream_id = f"{probe}.{stream}"
        segment_recordings = []

        for i, (_, srow) in enumerate(sframe.iterrows()):

            if time_ranges is not None:
                start_frame = int(time_ranges[i][0] * float(srow["imSampRate"]))
                end_frame = int(time_ranges[i][1] * float(srow["imSampRate"]))
            else:
                start_frame = srow["segmentFileStartSample"]
                end_frame = srow["segmentFileEndSample"]

            rec = load_single_segment_sglx_recording(
                srow.gate_dir, 
                srow.gate_dir_trigger_file_idx,
                stream_id,
                start_frame=start_frame,
                end_frame=end_frame,
            )

            segment_recordings.append(rec)

        # Sanity check
        for i, srec in enumerate(segment_recordings):
            if artifacts_frame is not None:
                npt.assert_almost_equal(srec.get_total_duration(), float(sframe["segmentTimeSecs"].values[i]), decimal=6)
                # assert srec.get_total_duration() == float(sframe["segmentTimeSecs"].values[i])
                assert srec.get_total_samples() == int(sframe["nSegmentSamp"].values[i])
            elif time_ranges is not None:
                npt.assert_almost_equal(srec.get_total_duration(), (time_ranges[i][1] - time_ranges[i][0]), decimal=3)
            else:
                # assert srec.get_total_duration() == float(sframe["fileTimeSecs"].values[i])  # Sometimes fileTimeSecs doesn't match nFileSamp
                assert srec.get_total_samples() == int(sframe["nFileSamp"].values[i])

        return segment_recordings

    def get_single_segment_si_recording(
        self,
        experimentName : str,
        aliasName : str,
        stream : str,
        probe : str,
        time_ranges : Optional[list] = None,
        artifacts_frame : Optional[pd.DataFrame] = None,
        sampling_frequency_max_diff : Optional[float] = 0,
    ): 
        single_segment_recordings = self._get_single_segment_si_recordings_list(
            experimentName,
            aliasName,
            stream,
            probe,
            time_ranges=time_ranges,
            artifacts_frame=artifacts_frame,
        )
        return si.concatenate_recordings(
            single_segment_recordings,
            sampling_frequency_max_diff=sampling_frequency_max_diff,
        )

    def get_multi_segment_si_recording(
        self,
        experimentName : str,
        aliasName : str,
        stream : str,
        probe : str,
        time_ranges : Optional[list] = None,
        artifacts_frame : Optional[pd.DataFrame] = None,
        sampling_frequency_max_diff : Optional[float] = 0,
    ):
        single_segment_recordings = self._get_single_segment_si_recordings_list(
            experimentName,
            aliasName,
            stream,
            probe,
            time_ranges=time_ranges,
            artifacts_frame=artifacts_frame,
        )
        return si.append_recordings(
            single_segment_recordings,
            sampling_frequency_max_diff=sampling_frequency_max_diff,
        )

    @staticmethod
    def load_yaml_doc(yaml_path):
        """Load a YAML file that contains only one document."""
        with open(yaml_path) as fp:
            yaml_doc = yaml.safe_load(fp)
        return yaml_doc


class SubjectLibrary:
    def __init__(self, libdir: Path):
        self.libdir = libdir
        self.cachefile = self.libdir / "wne_sglx_cache.pkl"
        self.cache = (
            pd.read_pickle(self.cachefile) if self.cachefile.is_file() else None
        )

    def get_subject_file(self, subjectName: str) -> Path:
        return self.libdir / f"{subjectName}.yml"

    def get_subject(self, subjectName: str) -> Subject:
        subjectFrame = (
            self.cache[self.cache["subject"] == subjectName]
            .drop(columns="subject")
            .reset_index(drop=True)
            if self.cache is not None
            else None
        )
        return Subject(self.get_subject_file(subjectName), subjectFrame)

    def get_subject_names(self) -> list[str]:
        return [f.stem for f in self.libdir.glob("*.yml")]

    def refresh_cache(self) -> pd.DataFrame:
        names = self.get_subject_names()
        subjectCaches = [Subject(self.get_subject_file(name)).cache for name in names]
        self.cache = pd.concat(
            subjectCaches, keys=names, names=["subject"]
        ).reset_index(level=0)
        return self.cache

    # TODO: This would be better written as an HTSV or PQT, with datatypes assigned during loading.
    def write_cache(self):
        if self.cache is None:
            self.refresh_cache()
        self.cache.to_pickle(self.cachefile)
