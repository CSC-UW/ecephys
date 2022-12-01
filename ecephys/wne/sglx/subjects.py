import logging
import yaml
import pandas as pd
import ecephys as ece
import spikeinterface as si
from pathlib import Path
from ecephys.utils.spikeinterface_utils import load_single_segment_sglx_recording

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

    # SpikeInterface loaders

    def get_si_single_segments(
        self,
        experiment,
        alias,
        stream,
        probe,
        time_ranges=None,
    ):
        ftable = self.get_file_frame(
            experiment, alias, probe=probe, stream=stream, ftype="bin"
        )
        stream_id = f"{probe}.{stream}"
        if time_ranges is not None:
            assert len(time_ranges) == len(ftable)
        single_segment_recordings = []
        for i, (_, row) in enumerate(ftable.iterrows()):
            rec = load_single_segment_sglx_recording(
                row.gate_dir, row.gate_dir_trigger_file_idx, stream_id
            )
            if time_ranges is not None:
                segment_time_range = time_ranges[i]
                rec = rec.frame_slice(
                    int(segment_time_range[0] * rec.get_sampling_frequency()),
                    int(segment_time_range[1] * rec.get_sampling_frequency()),
                )
                print(f"Add segment: time_range={segment_time_range}", rec)
            single_segment_recordings.append(rec)
        return single_segment_recordings

    def get_single_segment_si_recording(
        self,
        experiment,
        alias,
        stream,
        probe,
        sampling_frequency_max_diff=0,
        time_ranges=None,
    ):
        single_segment_recordings = self.get_si_single_segments(
            experiment,
            alias,
            stream,
            probe,
            time_ranges=time_ranges,
        )
        return si.concatenate_recordings(
            single_segment_recordings,
            sampling_frequency_max_diff=sampling_frequency_max_diff,
        )

    def get_multi_segment_si_recording(
        self,
        experiment,
        alias,
        stream,
        probe,
        sampling_frequency_max_diff=0,
        time_ranges=None,
    ):
        single_segment_recordings = self.get_si_single_segments(
            experiment,
            alias,
            stream,
            probe,
            time_ranges=time_ranges,
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
