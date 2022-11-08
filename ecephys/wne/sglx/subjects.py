import logging
from pathlib import Path

import ecephys as ece
import pandas as pd
import spikeinterface.extractors as se
import spikeinterface as si
import yaml

from ecephys.wne.sglx.spikeinterface_utils import load_single_segment_sglx_recording

logger = logging.getLogger(__name__)


class SubjectLibrary:
    def __init__(self, subjects_directory):
        self.libdir = Path(subjects_directory)

    def get_subject_file(self, subject_name):
        return self.libdir / f"{subject_name}.yml"

    def get_subject_document(self, subject_name):
        return SubjectLibrary.load_yaml_doc(self.get_subject_file(subject_name))

    def get_subject(self, subject_name):
        return Subject(subject_name, self.get_subject_document(subject_name))

    @staticmethod
    def load_yaml_doc(yaml_path):
        """Load a YAML file that contains only one document."""
        with open(yaml_path) as fp:
            yaml_doc = yaml.safe_load(fp)
        return yaml_doc


class Subject:
    def __init__(self, subject_name, subject_document):
        self.name = subject_name
        self.doc = subject_document

    def __repr__(self):
        return f"wneSubject: {self.name}"

    def get_alias_datetimes(self, experiment_name, alias_name):
        experiment = self.doc["experiments"][experiment_name]
        alias = experiment["aliases"][alias_name]

        def get_subalias_datetimes(subalias):
            if not ("start_time" in subalias) and ("end_time" in subalias):
                raise NotImplementedError(
                    f"All subaliases of {alias_name} must have start_time and end_time fields."
                )
            return (
                pd.to_datetime(subalias["start_time"]),
                pd.to_datetime(subalias["end_time"]),
            )

        return [get_subalias_datetimes(subalias) for subalias in alias]

    def get_files_table(
        self,
        experiment_name,
        alias_name=None,
        assert_contiguous=False,
        **kwargs,
    ):
        """Get all SpikeGLX files matching selection criteria.

        Parameters:
        -----------
        subject_name: string
        experiment_name: string
        alias_name: string (default: None)
        assert_contiguous: bool (default: False). If True and an alias was
            specified, assert that the all of the subaliases' start and end files
            were found, and that all the files are contiguous (enough) in between.

        Returns:
        --------
        pd.DataFrame:
            All requested files in sorted order.
        """
        sessions = self.doc["recording_sessions"]
        experiment = self.doc["experiments"][experiment_name]

        df = (
            ece.wne.sglx.experiments.get_alias_files_table(
                sessions, experiment, experiment["aliases"][alias_name]
            )
            if alias_name
            else ece.wne.sglx.experiments.get_experiment_files_table(
                sessions, experiment
            )
        )
        files_df = ece.sglx.loc(df, **kwargs)

        # TODO: This would be better as a check performed by the user on the returned DataFrame,
        # especially if they might want to use a tolerance of more than a sample or two.
        if assert_contiguous:
            assert files_df["isContinuation"][
                1:
            ].all(), "Files are not contiguous to within the specified tolerance."

        return files_df.reset_index(drop=True)

    def get_lfp_bin_paths(self, experiment, alias=None, **kwargs):
        return self.get_files_table(
            experiment, alias, stream="lf", ftype="bin", **kwargs
        ).path.values

    def get_ap_bin_paths(self, experiment, alias=None, **kwargs):
        return self.get_files_table(
            experiment, alias, stream="ap", ftype="bin", **kwargs
        ).path.values

    def get_lfp_bin_table(self, experiment, alias=None, **kwargs):
        return self.get_files_table(
            experiment, alias, stream="lf", ftype="bin", **kwargs
        )

    def get_ap_bin_table(self, experiment, alias=None, **kwargs):
        return self.get_files_table(
            experiment, alias, stream="ap", ftype="bin", **kwargs
        )

    def get_experiment_data_times(self, experiment, probe, as_datetimes=False):
        fTable = self.get_files_table(experiment, probe=probe)
        if as_datetimes:
            return (fTable.wneFileStartDatetime.min(), fTable.wneFileEndDatetime.max())
        else:
            return (fTable.wneFileStartTime.min(), fTable.wneFileEndTime.max())

    def t2dt(self, experiment, probe, t):
        dt0, _ = self.get_experiment_data_times(experiment, probe, as_datetimes=True)
        return pd.to_timedelta(t, "s") + dt0

    def dt2t(self, experiment, probe, dt):
        dt0, _ = self.get_experiment_data_times(experiment, probe, as_datetimes=True)
        return (dt - dt0) / pd.to_timedelta("1s")

    # SpikeInterface loaders

    def get_si_single_segments(
        self,
        experiment,
        alias,
        stream,
        probe,
        time_ranges=None,
    ):
        ftable = self.get_files_table(
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
            single_segment_recordings.append(
                rec
            )
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
