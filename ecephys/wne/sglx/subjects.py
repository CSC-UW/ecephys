import logging
from pathlib import Path

import ecephys as ece
import pandas as pd
import spikeinterface.extractors as se
import yaml

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

        return files_df

    # TODO: Values should already be sorted by fileCreateTime. Check that the sort here is unnecessary
    def get_lfp_bin_paths(self, experiment, alias=None, **kwargs):
        return (
            self.get_files_table(experiment, alias, stream="lf", ftype="bin", **kwargs)
            .sort_values("fileCreateTime", ascending=True)
            .path.values
        )

    # TODO: Values should already be sorted by fileCreateTime. Check that the sort here is unnecessary
    def get_ap_bin_paths(self, experiment, alias=None, **kwargs):
        return (
            self.get_files_table(experiment, alias, stream="ap", ftype="bin", **kwargs)
            .sort_values("fileCreateTime", ascending=True)
            .path.values
        )

    # TODO: Values should already be sorted by fileCreateTime. Check that the sort here is unnecessary
    def get_lfp_bin_table(self, experiment, alias=None, **kwargs):
        return self.get_files_table(
            experiment, alias, stream="lf", ftype="bin", **kwargs
        ).sort_values("fileCreateTime", ascending=True)

    # TODO: Values should already be sorted by fileCreateTime. Check that the sort here is unnecessary
    def get_ap_bin_table(self, experiment, alias=None, **kwargs):
        return self.get_files_table(
            experiment, alias, stream="ap", ftype="bin", **kwargs
        ).sort_values("fileCreateTime", ascending=True)

    def get_alias_start(self, experiment, alias, probe):
        fTable = self.get_files_table(experiment, alias, probe=probe)
        return fTable.tExperiment.min(), fTable.dtExperiment.min()

    def get_experiment_start(self, experiment, probe):
        return self.get_alias_start(experiment, alias=None, probe=probe)

    def t2dt(self, experiment, probe, t):
        _, dt0 = self.get_experiment_start(experiment, probe)
        return pd.to_timedelta(t, "s") + dt0
