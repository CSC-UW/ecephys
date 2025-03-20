import itertools as it
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ecephys.sglx import file_mgmt
from ecephys.wne.sglx import experiments, sessions
from ecephys.wne.subject import Subject

logger = logging.getLogger(__name__)


class SGLXSubject(Subject):
    """The cache contains the sessions frame."""

    def __init__(
        self, subjectYamlFile: Path, subjectCache: Optional[pd.DataFrame] = None
    ):
        Subject.__init__(self, subjectYamlFile)
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
            if "start_time" not in subalias and ("end_time" in subalias):
                raise NotImplementedError(
                    f"All subaliases of {alias} must have start_time and end_time fields."
                )
            return (
                pd.to_datetime(subalias["start_time"]),
                pd.to_datetime(subalias["end_time"]),
            )

        return [get_subalias_datetimes(sa) for sa in subaliases]

    def get_tdt_block_path(self, experiment: str):
        return Path(self.doc["experiments"][experiment]["tdt_block_path"])


class SGLXSubjectLibrary:
    def __init__(self, libdir: Path):
        self.libdir = libdir
        self.cachefile = (
            self.libdir / "wne_sglx_cache.gz"
        )  # TODO: Extension should be .pkl.gz
        self.cache = self.read_cache() if self.cachefile.is_file() else None

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
    # We compress the pickle using GZIP because we exceeded GitHub's filesize limit. This is a bad system.
    def write_cache(self):
        if self.cache is None:
            self.refresh_cache()
        self.cache.to_pickle(self.cachefile, compression="gzip")

    def read_cache(self):
        return pd.read_pickle(self.cachefile, compression="gzip")


# TODO: Remove as soon as SpikeInterface adds this functionality.
# 1/30/2023 Tom says It is still necessary, because although the relevant GitHub issues seem to have been closed, the API has not changed.
# To me, it looks like "gate_dir_trigger_file_idx" is same or similar to "seg_index" from neo.rawio.SpikeGLXRawIO.scan_files.
# Indeed, this function does basically get the segment index into an extractor created later, and could even be named
# add_si_segment_indices(). I think what we should do, in order to get rid of this function, is to use the neo header in
# the SI extractor to add the segment indices to a ftab, rather than trying to anticipate them.
def _get_gate_dir_trigger_file_index(ftab: pd.DataFrame) -> pd.DataFrame:
    """Get index of trigger file relative to all files of same stream/prb/gate_folder.

    This is relative to files currently present in the directory, so we can't
    just parse trigger index (doesn't work if some files are moved).

    Useful to instantiate spikeinterface extractor objects, since they require
    subselecting segments of interest (ie trigger files) after instantiation.
    See https://github.com/SpikeInterface/spikeinterface/issues/628#issuecomment-1130232542

    """  # This actually seems like a more appropriate issue: https://github.com/NeuralEnsemble/python-neo/pull/1125#issuecomment-1148930135
    ftab["gate_dir"] = ftab.apply(
        lambda row: row["path"].parent, axis=1
    )  # TODO: This is WRONG! This yields the probe directory, not the gate directory.
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
