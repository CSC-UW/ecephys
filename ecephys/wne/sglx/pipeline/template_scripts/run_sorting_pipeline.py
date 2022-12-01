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
  --rerun_existing                   Rerun things
  --opts_dirpath==<odp>              Path to directory containing option file [default: {OPTS_DIRPATH}]
  --opts_filename==<ofn>             Name of options file [default: {OPTS_FILENAME}]
  --projectName==<pn>                Project name [default: {PROJECT_NAME}]
  --experimentName==<en>             Exp name [default: {EXPERIMENT_NAME}]
  --aliasName==<an>                  Alias name [default: {ALIAS_NAME}]
  --output_dirname==<dn>             Name of output dir. Pulled from opts file if None. [default: {OUTPUT_DIRNAME}]

"""
from pathlib import Path

from docopt import docopt

import wisc_ecephys_tools as wet
from ecephys.wne.sglx.pipeline.sorting_pipeline import \
    SpikeInterfaceSortingPipeline

subjectsDir = wet.get_subjects_directory()
projectsFile = wet.get_projects_file()

DEFAULT_VALUES = {
  "PROJECT_NAME": "my_project",
  "EXPERIMENT_NAME": "my_experiment",
  "ALIAS_NAME": "my_alias",
  "OPTS_DIRPATH": "/path/to/opts/such/as/ecephys/wne/sglx/pipeline/template_opts",
  "OPTS_FILENAME": "template_pipeline_opts.yaml",
  "OUTPUT_DIRNAME": None,
}

TIME_RANGES=None

if __name__ == "__main__":

    args = docopt(__doc__.format(**DEFAULT_VALUES), version='Naval Fate 2.0')

    opts_filepath = Path(args["--OPTS_DIRPATH"])/args["--OPTS_FILENAME"]

    sorting_pipeline = SpikeInterfaceSortingPipeline(
        args["<subjectName>"],
        subjectsDir,
        args["--projectName"],
        projectsFile,
        args["--experimentName"],
        args["--aliasName"],
        args["<probeName>"],
        time_ranges=TIME_RANGES,
        opts_filepath=opts_filepath,
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
