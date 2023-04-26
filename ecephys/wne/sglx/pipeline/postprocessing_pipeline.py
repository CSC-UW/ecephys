import deepdiff
from horology import Timing
import json
import logging
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from pathlib import Path
import probeinterface as pi
from typing import Optional, Union
from tqdm import tqdm
import textwrap
import yaml
import spikeinterface.extractors as se
import spikeinterface.full as si
import spikeinterface.postprocessing as sp
import spikeinterface.qualitymetrics as sq
import spikeinterface.sorters as ss
from spikeinterface.qualitymetrics.misc_metrics import compute_amplitude_cutoffs

from .sorting_pipeline import SpikeInterfaceSortingPipeline, get_main_output_dir
from ..subjects import Subject
from ...projects import Project
from ... import constants
from ...utils import load_hypnogram_for_si_slicing
from .... import utils as ece_utils
from .... import hypnogram as hp

from ecephys.utils.siutils import cut_and_combine_si_extractors

# TODO: Be consistent about using logger vs print
logger = logging.getLogger(__name__)

Pathlike = Union[Path, str]
POSTPRO_OPTS_FNAME = "postpro_opts.yaml"
OUTPUT_HYPNO_FNAME = "hypnogram.htsv"

HYPNOGRAM_IGNORED_STATES = [
    "Other",
    "IS",
    "Artifact",
    "MA",
    None,
    "None",
]

METRICS_COLUMNS_TO_PLOT = [
    "group", "firing_rate", "presence_ratio", "snr",
    "isi_violations_ratio", "rp_contamination", "sliding_rp_violation",
    "amplitude_cutoff", "drift_ptp", "nn_isolation", "nn_noise_overlap",
    "nn_unit_id"
]

"""
- <project>
    - <experiment>
        - <alias>
            - <subject>
                - <sorting_basename>.<probe>/ # This is the main output directory
                    - si_output/ # This is Spikeinterface's output directory. Everything in there is deleted by spikeinterface when running run_sorter.
                        - sorter_output/ # This is the sorter output directory
                    - preprocessing # This is the preprocessing output directory
                    - <basename> # This is the postprocessing output directory
                        - si_output_full/ # This is Spikeinterface's output ("waveforms") directory for the whole recording. Everything in there may be deleted by spikeinterface
                        - si_output_<vigilance_state_1>/ # Waveform directory for vigilance_state_1
                        - si_output_<vigilance_state_2>/ # Waveform directory for vigilance_state_2
                        ...
"""


# Hacking around Spikeinterface: Load kilosort `amplitudes.npy` file directly
# to compute amplitude_cutoff metric, since this metric doesn't work
# With the amplitudes available in SI.
def _load_kilosort_amplitudes_by_cluster(we, ks_dir):

    # Only for full sorting
    assert not isinstance(we.sorting, (si.ConcatenateSegmentSorting, si.FrameSliceSorting))
    assert isinstance(we.sorting, (si.KiloSortSortingExtractor, si.PhySortingExtractor))

    spike_amplitudes = np.load(ks_dir/"amplitudes.npy")
    spike_unit_ids = np.load(ks_dir/"spike_clusters.npy")

    return [{
        unit_id: np.atleast_1d(spike_amplitudes[np.where(spike_unit_ids == unit_id)[0]].squeeze())
        for unit_id in np.unique(spike_unit_ids)
    }]


class SpikeInterfacePostprocessingPipeline:
    def __init__(
        self,
        wneProject: Project,
        wneSubject: Subject,
        experiment: str,
        alias: str,
        probe: str,
        sorting_basename: str,
        postprocessing_name: str = "postprocessing_df",
        rerun_existing: bool = False,
        n_jobs: int = 1,
        options_source: Union[str, Path] = "wneProject",
        hypnogram_source: Union[str, Path, Project] = None,
    ):
        # These properties are private, because they should not be modified after instantiation
        self._wneProject = wneProject
        self._wneSubject = wneSubject
        self._alias = alias
        self._experiment = experiment
        self._probe = probe
        self._postprocessing_name = postprocessing_name
        self._sorting_basename = sorting_basename
        self._rerun_existing = rerun_existing
        self._nJobs = int(n_jobs)

        self._sorting_pipeline = SpikeInterfaceSortingPipeline.load_from_folder(
            self._wneProject,
            self._wneSubject,
            self._experiment,
            self._alias,
            self._probe,
            self._sorting_basename,
            rerun_existing=False,
        )

        # Set the location where options should be loaded from. Either...
        #   (1) wneProject
        #   (2) A custom location of your choosing
        if options_source == "wneProject":
            self._opts_src = wneProject.get_experiment_subject_file(
                experiment,
                wneSubject.name,
                constants.PREPROCESSING_PIPELINE_PARAMS_FNAME,
            )
        else:
            self._opts_src = Path(options_source)

        # Load the options, either using JSON or YAML (for backwards compatibility)
        with open(self._opts_src, "r") as f:
            suffix = self._opts_src.suffix
            if suffix == ".json":
                self._opts = json.load(f)
            elif suffix in {".yaml", ".yml"}:
                self._opts = yaml.load(f, Loader=yaml.SafeLoader)
            else:
                raise ValueError(
                    f"Unexpected file extension {suffix} on {self._opts_src}"
                )
        # Ensure we use same opts as previously
        prior_opts_path = self.postprocessing_output_dir / POSTPRO_OPTS_FNAME
        if prior_opts_path.exists():
            with open(prior_opts_path, "r") as f:
                prior_opts = yaml.load(f, Loader=yaml.SafeLoader)
            if deepdiff.DeepDiff(self._opts, prior_opts):
                raise ValueError(
                    f"Current options do not match prior options saved on file at {prior_opts_path}.\n"
                    f"Current opts: {self._opts}\n"
                    f"Prior opts: {prior_opts}\n"
                )

        # Pull hypnogram if provided
        if hypnogram_source is None:
            self._hypnogram = None
            self._hypnogram_states = None
        else:
            if isinstance(hypnogram_source, Project):
                self._hypnogram = load_hypnogram_for_si_slicing(
                    hypnogram_source,
                    self._wneProject,
                    self._wneSubject,
                    self._experiment,
                    self._alias,
                    self._probe,
                    self._sorting_basename,
                    precision_s=0.01,
                    allow_no_sync_file=True,
                    simplify=True,
                ).drop_states(HYPNOGRAM_IGNORED_STATES)._df.reset_index(drop=True)
            elif isinstance(hypnogram_source, (str, Path)):
                # Only when instantiating with load_from_folder
                prior_hypno_path = self.postprocessing_output_dir / OUTPUT_HYPNO_FNAME
                assert prior_hypno_path == hypnogram_source
                self._hypnogram = ece_utils.read_htsv(hypnogram_source)
            else:
                raise ValueError(f"Unrecognize type for `hypnogram_source`: {hypnogram_source}")
            self._hypnogram_states = self._hypnogram.state.unique()
            assert not None in self._hypnogram_states

        # Ensure we use same hypno as previously
        prior_hypno_path = self.postprocessing_output_dir / OUTPUT_HYPNO_FNAME
        if prior_hypno_path.exists():
            prior_hypno = ece_utils.read_htsv(prior_hypno_path)
            try:
                assert_frame_equal(
                    prior_hypno, self._hypnogram, check_dtype=False
                )
            except AssertionError as e:
                raise ValueError(
                    f"New hypnogram doesn't match previously used hypnogram file at {prior_hypno_path}:\n"
                    f"Consider instantiating pipeline object with `SpikeinterfacePostprocessingPipeline.load_from_folder`\n"
                    f"{e}"
                )

        # Run postprocessing state_by_state:
        self._run_by_state = False
        self._opts_by_state = None
        if "by_hypnogram_state" in self._opts:
            self._run_by_state = True
            self._opts_by_state = self._opts["by_hypnogram_state"]
            assert {"waveforms", "postprocessing", "metrics"}.issubset(self._opts_by_state)
            assert self._hypnogram is not None, "Opts request processing by state, but no hypnogram was found"

        # Set compute details
        self.job_kwargs = {
            "n_jobs": self._nJobs,
            "chunk_duration": "1s",
            "progress_bar": True,
        }

        # Cached waveform extractors
        # Keys are vigilance state. `None` entry is the full recording
        self._si_waveform_extractor_by_state = {}

        self.check_sorting_output()
    
    def check_sorting_output(self):
        assert self._sorting_pipeline.is_sorted
        # TODO: This shouldn't be necessary... 
        # But I noticed that before opening/saving with phy, the returned sorter may have clusters
        # with N=0 spikes, which messes up postprocessing. Those are filtered out somehow when running phy.
        assert (self._sorting_pipeline.sorter_output_dir / "cluster_info.tsv").exists(), (
            f"You need to open/save this sorting with Phy first."
        )
    
    def __repr__(self):
        repr = f"""
        {self.__class__}
        Postprocessing output dir: {self.postprocessing_output_dir}
        Options source: {self._opts_src}
        """
        if self._hypnogram is not None:
            repr += f"""Hypnogram: {self._hypnogram.groupby('state')["duration"].sum()}"""
        else:
            repr += f"Hypnogram: None"

        return repr + f"\n\nSorting pipeline: {self._sorting_pipeline}"

    # Paths

    @property
    def main_output_dir(self) -> Path:
        return self._sorting_pipeline.main_output_dir

    @property
    def sorter_output_dir(self) -> Path:
        return self._sorting_pipeline.sorter_output_dir

    @property
    def postprocessing_output_dir(self) -> Path:
        return self.main_output_dir / self._postprocessing_name
    
    @property
    def summary_plots_output_dir(self) -> Path:
        return self.postprocessing_output_dir/"summary_plots"
    
    # State dependant

    def is_postprocessed_by_state(self, state=None) -> bool:
        return (
            self._sorting_pipeline.is_sorted
            and self.get_waveforms_output_dir_by_state(state=state).exists()
            and (
                self.get_waveforms_output_dir_by_state(state=state) / "quality_metrics" / "metrics.csv"
            ).exists()
        )

    def get_waveforms_output_dir_by_state(self, state=None) -> Path:
        """Below postprocessing_output_dir because content deleted by spikeinterface"""
        dirname = "si_output"
        if state is not None:
            dirname += f"_{state}"
        return self.postprocessing_output_dir / dirname

    def get_waveform_extractor_by_state(self, state=None) -> si.WaveformExtractor:
        """Return cached or load/extract (full or state-specific) waveform extractor."""
        if self._si_waveform_extractor_by_state.get(state, None) is None:
            self._si_waveform_extractor_by_state[state] = self.load_or_extract_waveform_extractor_by_state(state=state)
        return self._si_waveform_extractor_by_state[state]

    def get_recording_for_waveforms_by_state(self, state=None) -> si.BaseRecording:
        """Return cached or load/extract (full or state-specific) waveform extractor."""
        rec = self._sorting_pipeline.processed_extractor_for_waveforms

        if state is None:
            return rec
        
        return cut_and_combine_si_extractors(
            rec,
            self._hypnogram[self._hypnogram["state"] == state].copy(),
            combine="concatenate",
        )

    def get_sorting_for_waveforms_by_state(self, state=None) -> si.BaseSorting:
        """Return cached or load/extract (full or state-specific) waveform extractor."""
        sorting = self._sorting_pipeline.sorting_extractor

        if state is None:
            return sorting

        return cut_and_combine_si_extractors(
            sorting, 
            self._hypnogram[self._hypnogram["state"] == state],
            combine="concatenate"
        )
    
    def get_opts_by_state(self, state=None):
        if state is None:
            return self._opts.copy()

        return self._opts_by_state.copy()

    def load_or_extract_waveform_extractor_by_state(self, state=None, opts=None) -> si.WaveformExtractor:
        """Load a waveform extractor. This may `extract` previously computed and saved waveforms."""

        opts = self.get_opts_by_state(state=state)

        if not "waveforms" in opts:
            raise ValueError(f"Expected 'waveforms' entry in postprocessing options: {opts}.")
        waveforms_kwargs = opts["waveforms"]

        assert (
            self._sorting_pipeline.is_sorted
        ), "Cannot load waveform extractor for unsorted recording."

        waveform_recording = self.get_recording_for_waveforms_by_state(state=state)
        waveform_sorting = self.get_sorting_for_waveforms_by_state(state=state)
        waveform_output_dir = self.get_waveforms_output_dir_by_state(state=state) 
        assert waveform_recording.get_total_samples() == waveform_sorting.get_total_samples()

        # If we have already extracted waveforms before, re-use those.
        load_precomputed_waveforms = not self._rerun_existing and self.is_postprocessed_by_state(state=state)
        if load_precomputed_waveforms:
            print(
                f"Loading precomputed waveforms. Will use this recording: {waveform_recording}"
            )
            we = si.WaveformExtractor.load_from_folder(
                waveform_output_dir,
                with_recording=False,
                sorting=waveform_sorting,
            )
            # Hack to have recording even if we deleted or moved the data,
            # because load_from_folder(.., with_recording=True) assumes same recording file locations etc
            we._recording = waveform_recording
        else:
            with Timing(name="Extract waveforms: "):
                we = si.extract_waveforms(
                    waveform_recording,
                    waveform_sorting,
                    folder=waveform_output_dir,
                    overwrite=True,
                    **waveforms_kwargs,
                    **self.job_kwargs,
                )

        return we

    def _run_si_metrics_by_state(self, state=None):

        opts = self.get_opts_by_state(state=state)

        if not "metrics" in opts:
            raise ValueError(f"Expected 'metrics' entry in postprocessing options: {opts}")
        metrics_opts = opts["metrics"]

        we = self.get_waveform_extractor_by_state(state=state)

        run_amplitude_cutoffs = False
        if "amplitude_cutoff" in metrics_opts:
            if state is not None:
                raise ValueError(
                    "`amplitude_cutoff` metric is only available for full recording (not by state).\n"
                    "Please modify postpro opts"
                )
            print("Computing amplitude_cutoff metric separately")
            amplitude_cutoff_opts = metrics_opts.pop("amplitude_cutoff")
            kilosort_spike_amplitudes = _load_kilosort_amplitudes_by_cluster(
                we,
                self._sorting_pipeline.sorter_output_dir
            )
            amplitude_cutoff_res = compute_amplitude_cutoffs(
                we,
                spike_amplitudes = kilosort_spike_amplitudes,
                **amplitude_cutoff_opts,
            )
            run_amplitude_cutoffs = True

        # Set job kwargs for all
        si.set_global_job_kwargs(n_jobs=self._nJobs)

        metrics_names = list(metrics_opts.keys())
        print(f"Running metrics: {metrics_names}")
        print(f"Metrics params: {metrics_opts}")

        print("Computing metrics")
        with Timing(name="Compute metrics: "):
            metrics = sq.compute_quality_metrics(
                we,
                metric_names=metrics_names,
                qm_params=metrics_opts,
                progress_bar=True,
                verbose=True,
            )

        if run_amplitude_cutoffs:
            metrics["amplitude_cutoff"] = pd.Series(amplitude_cutoff_res)

        return metrics

    def _run_si_postprocessing_by_state(self, state=None):

        opts = self.get_opts_by_state(state=state)

        if not "postprocessing" in opts:
            raise ValueError("Expected 'postprocessing' entry in postprocessing options: {opts}")
        postprocessing_opts = opts["postprocessing"]

        # Set job kwargs for all
        si.set_global_job_kwargs(n_jobs=self._nJobs)

        # Iterate on spikeinterface.postprocessing functions by name
        for func_name, func_kwargs in postprocessing_opts.items():

            if not hasattr(sp, func_name):
                raise ValueError(
                    "Could not find function `{func_name}` in spikeinterface.postprocessing module"
                )
            func = getattr(sp, func_name)

            with Timing(name=f"{func_name}: "):
                func(
                    self.get_waveform_extractor_by_state(state=state),
                    load_if_exists=True,
                    **func_kwargs,
                )


    def run_postprocessing(self):

        self.postprocessing_output_dir.mkdir(exist_ok=True)
        # Save hypnogram used
        if self._hypnogram is not None:
            ece_utils.write_htsv(
                self._hypnogram,
                self.postprocessing_output_dir / OUTPUT_HYPNO_FNAME
            )

        # Save options used
        with open(self.postprocessing_output_dir / POSTPRO_OPTS_FNAME, "w") as f:
            yaml.dump(self._opts, f)

        # Whole recording
        print("Run full recording")
        self._run_si_postprocessing_by_state(state=None)
        all_states_metrics = self._run_si_metrics_by_state(state=None)
        print("\n\n")

        if self._run_by_state:
            for state in self._hypnogram_states:
                print(f"Run state={state}")
                self._run_si_postprocessing_by_state(state=state)
                state_metrics = self._run_si_metrics_by_state(state=state)
                # Check same cluster ids
                assert state_metrics.index.equals(all_states_metrics.index)
                # Append state to column names and add to aggregate array
                state_metrics.columns = [c + f"_{state}" for c in state_metrics.columns]
                all_states_metrics = all_states_metrics.join(state_metrics)
                print("\n\n")

        # Save aggregated metrics
        all_states_metrics.to_csv(
            self.postprocessing_output_dir/"metrics.csv",
            index=True,
            index_label="cluster_id"
        )

        # Postprocessing summary
        self.plot_summary()

        print("Done postprocessing.")

    ### Load and plot output

    def load_metrics(self):
        assert (self.postprocessing_output_dir/"metrics.csv").exists()
        metrics = pd.read_csv(
            self.postprocessing_output_dir/"metrics.csv"
        )

        # Add group
        cluster_group_path = self._sorting_pipeline.sorter_output_dir/"cluster_group.tsv"
        assert cluster_group_path.exists()
        cluster_group = pd.read_csv(
            cluster_group_path, sep="\t"
        )
        # Add cluster with "unsorted" group if missing
        for cluster_id in metrics.cluster_id.unique():
            if cluster_id not in cluster_group.cluster_id:
                cluster_group.loc[len(cluster_group), ['cluster_id','group']] = cluster_id, "unsorted"
        
        return pd.merge(
            metrics,
            cluster_group,
            on="cluster_id",
        )

    def plot_summary(self):

        self.summary_plots_output_dir.mkdir(exist_ok=True)
        self.plot_metrics_distributions()
        self.plot_unit_summaries()

    def plot_metrics_distributions(self):
        
        import seaborn as sns
        import matplotlib.pyplot as plt
        from pandas.api.types import is_numeric_dtype

        output_dir = self.summary_plots_output_dir
        output_dir.mkdir(exist_ok=True, parents=True)

        metrics = self.load_metrics()

        select_cols = [c for c in metrics.columns if any([n in c for n in METRICS_COLUMNS_TO_PLOT])]

        print(f"Plot metrics distribution")
        for name in tqdm(select_cols):

            if not is_numeric_dtype(metrics[name]):
                continue
            if metrics[name].isna().all():
                continue

            if metrics[name].max() > 1:
                xlim=(0, metrics[name].quantile(0.95))
            else:
                xlim=(0, metrics[name].max())
                
            title = f"{name}. NaNs={metrics[name].isna().sum()}/{len(metrics)}"
                
            p = sns.jointplot(data=metrics, x=name, y="firing_rate", hue="group", xlim=xlim)
            
            p.fig.suptitle(title)
            p.fig.tight_layout()
            
            p.fig.savefig(output_dir/f"metric_{name}.png")
            plt.clf()

    def plot_unit_summaries(self):

        import spikeinterface.widgets as sw
        import matplotlib.pyplot as plt

        output_dir = self.summary_plots_output_dir/"units"
        output_dir.mkdir(exist_ok=True, parents=True)
        
        metrics = self.load_metrics().copy().set_index("cluster_id")
        we = self.get_waveform_extractor_by_state()

        select_cols = [c for c in metrics.columns if any([n in c for n in METRICS_COLUMNS_TO_PLOT])]

        print(f"Plot unit summaries")

        for unit_id in tqdm(we.sorting.get_unit_ids()):
            w = sw.plot_unit_summary(we, unit_id=unit_id)
            
            # select a row by index and convert it to a dictionary
            row_dict = metrics.loc[unit_id].to_dict()
            # create a list comprehension of strings "<column>: <value>"
            # round float values if the column is in select_metrics
            title_parts = [f"unit_id: {unit_id}"]
            for col in select_cols:
                val = row_dict[col]
                if isinstance(val, float):
                    val = round(val, 3)  # round to 3 decimal places
                title_parts.append(f"{col}: {val}")
            title = textwrap.fill(", ".join(title_parts), width=100)

            w.figure.suptitle(title)
            
            w.figure.savefig(output_dir/f"unit_{unit_id}.png")
            plt.clf()

    ##### Reinstantiate #####

    @classmethod
    def load_from_folder(
        cls,
        wneProject: Project,
        wneSubject: Subject,
        experiment: str,
        alias: str,
        probe: str,
        sorting_basename: str,
        postprocessing_name: str,
        rerun_existing: bool = False,
    ):
        main_output_dir = get_main_output_dir(
            wneProject,
            wneSubject,
            experiment,
            alias,
            probe,
            sorting_basename,
        )
        postpro_output_dir = main_output_dir / postprocessing_name

        if not all(
            [
                f.exists()
                for f in [
                    postpro_output_dir,
                    postpro_output_dir / POSTPRO_OPTS_FNAME,
                ]
            ]
        ):
            raise FileNotFoundError(
                f"Could not find all required files in {postpro_output_dir}"
            )

        saved_hypnogram_path=(postpro_output_dir / OUTPUT_HYPNO_FNAME)
        if saved_hypnogram_path.exists():
            hypnogram_source = saved_hypnogram_path
        else:
            hypnogram_source = None

        return SpikeInterfacePostprocessingPipeline(
            wneProject,
            wneSubject,
            experiment,
            alias,
            probe,
            sorting_basename,
            postprocessing_name,
            rerun_existing=rerun_existing,
            options_source=(postpro_output_dir / POSTPRO_OPTS_FNAME),
            hypnogram_source=hypnogram_source,
        )
