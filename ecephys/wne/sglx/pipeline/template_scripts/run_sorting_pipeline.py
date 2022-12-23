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
  --n_jobs=<n_jobs>                  Number of jobs for all spikeinterface functions. [default: {N_JOBS}]

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
  "N_JOBS": 10,
}

TIME_RANGES=None

if __name__ == "__main__":

    args = docopt(__doc__.format(**DEFAULT_VALUES), version='Naval Fate 2.0')

    opts_filepath = Path(args["--OPTS_DIRPATH"])/args["--OPTS_FILENAME"]

    print(f"Running all subject/probe pairs: {args['--input']}")

    for subject_probe in args["--input"]:

      subjectName, probeName = subject_probe.split(",")

      sorting_pipeline = SpikeInterfaceSortingPipeline(
          subjectName,
          subjectsDir,
          args["--projectName"],
          projectsFile,
          args["--experimentName"],
          args["--aliasName"],
          probeName,
          time_ranges=TIME_RANGES,
          opts_filepath=opts_filepath,
          rerun_existing=args["--rerun_existing"],
          output_dirname=args["--output_dirname"],
          n_jobs=args["--n_jobs"],
      )
      print(f"Sorting pipeline: {sorting_pipeline}\n")
      print(f"Pipeline opts: {sorting_pipeline.opts}")
      print(f"Raw recording:\n {sorting_pipeline.raw_si_recording}")

      if args["--prepro_only"]:
          print("--prepro_only==True: Run only preprocessing")
          sorting_pipeline.run_preprocessing()
      elif args["--metrics_only"]:
          print("--metrics_only==True: Run only preprocessing")
          sorting_pipeline.run_metrics()
      else:
          print("--prepro_only==False: Run full pipeline")
          sorting_pipeline.run_pipeline()

      print(f"Sorting pipeline: {sorting_pipeline}\n")
      print(f"Pipeline opts: {sorting_pipeline.opts}\n")
      print(f"Raw recording:\n {sorting_pipeline.raw_si_recording}\n")

      print(f"\n\n ...Done!")