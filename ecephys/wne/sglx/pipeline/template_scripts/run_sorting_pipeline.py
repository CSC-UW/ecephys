#!/usr/bin/env python
# coding: utf-8

"""Run sorting pipeline.

Modify script to change default argument values

Usage:
  run_sorting_pipeline.py --help
  run_sorting_pipeline.py <subjectName> <probeName> [options]

Options:
  -h --help                          Show this screen.
  --prepro_only                      Run only preprocessing, not full pipeline (drift correction) [default False]
  --opts_filepath==<fp>              Path to options file [default: {OPTS_FILEPATH}]
  --projectName==<pn>                Project name [default: {PROJECT_NAME}]
  --experimentName==<en>             Exp name [default: {EXPERIMENT_NAME}]
  --aliasName==<an>                  Alias name [default: {ALIAS_NAME}]
  --rerun_existing==<re>             Rerun things [default: {RERUN_EXISTING}]
  --output_dirname==<dn>             Name of output dir. Pulled from opts file if None. [default: {OUTPUT_DIRNAME}]

"""
from docopt import docopt
from ecephys.wne.sglx.pipeline.sorting_pipeline import SortingPipeline, SpikeInterfaceSortingPipeline
import wisc_ecephys_tools as wet

subjectsDir = wet.get_subjects_directory()
projectsFile = wet.get_projects_file()

DEFAULT_VALUES = {
  "PROJECT_NAME": "my_project",
  "EXPERIMENT_NAME": "my_experiment",
  "ALIAS_NAME": "my_alias",
  "OPTS_FILEPATH": "/path/to/opts/such/as/ecephys/wne/sglx/pipeline/template_opts/template_pipeline_opts.yaml",
  "RERUN_EXISTING": False,
  "OUTPUT_DIRNAME": None,
}

TIME_RANGES=None

if __name__ == "__main__":

    args = docopt(__doc__.format(**DEFAULT_VALUES), version='Naval Fate 2.0')

    sorting_pipeline = SpikeInterfaceSortingPipeline(
        args["<subjectName>"],
        subjectsDir,
        args["--projectName"],
        projectsFile,
        args["--experimentName"],
        args["--aliasName"],
        args["<probeName>"],
        time_ranges=TIME_RANGES,
        opts_filepath=args["--opts_filepath"],
        rerun_existing=args["--rerun_existing"],
        output_dirname=args["--output_dirname"],
    )
    print(sorting_pipeline)

    print(f"Pipeline opts: {sorting_pipeline.opts}")
    print(f"Raw recording:\n {sorting_pipeline.raw_si_recording}")

    if args["--prepro_only"]:
        print("--prepro_only==True: Run only preprocessing")
        sorting_pipeline.run_preprocessing()
    else:
        print("--prepro_only==False: Run full pipeline")
        sorting_pipeline.run_pipeline()