#!/usr/bin/env python
# coding: utf-8

"""Run sorting pipeline.

Modify script to change default argument values

Usage:
  run_sorting_pipeline.py [options] [--input <subjectName>,<probeName>]...
  run_sorting_pipeline.py [options] (--prepro_only) [--input <subjectName>,<probeName>]...

Options:
  -h --help                          Show this screen.
  --input==<subjectName,probeName>   (Repeatable) Comma-separated pair of the form `<subjectName>,<probeName>`
  --prepro_only                      Run only preprocessing, not full pipeline (drift correction)
  --rerun_existing                   Rerun rather than load things.
  --options_source==<ofn>            Name of options file (applied to all input datasets) [default: {OPTIONS_SOURCE}] # TODO
  --projectName==<pn>                Project name [default: {PROJECT_NAME}]
  --experimentName==<en>             Exp name [default: {EXPERIMENT_NAME}]
  --aliasName==<an>                  Alias name [default: {ALIAS_NAME}]
  --basename==<bn>                   Name of output dir. [default: {BASENAME}]
  --exclusionsPath=<exclusions>      Path to exclusion file. Pulled from WNE project if unspecified.
  --n_jobs=<n_jobs>                  Number of jobs for all spikeinterface functions. [default: {N_JOBS}]

"""
from pathlib import Path

import pandas as pd
from docopt import docopt

import wisc_ecephys_tools as wet
from ecephys.wne.sglx.pipeline.sorting_pipeline import SpikeInterfaceSortingPipeline

DEFAULT_VALUES = {
    "PROJECT_NAME": "my_project",
    "EXPERIMENT_NAME": "my_experiment",
    "ALIAS_NAME": "my_alias",
    "OPTIONS_SOURCE": "/path/to/opts/such/as/ecephys/wne/sglx/pipeline/template_sorting_opts.yaml",
    "BASENAME": "sorting_df",
    "N_JOBS": 10,
}

if __name__ == "__main__":
    args = docopt(__doc__.format(**DEFAULT_VALUES), version="Naval Fate 2.0")

    print(f"Running all subject/probe pairs: {args['--input']}")

    for subject_probe in args["--input"]:
        subjectName, probe = subject_probe.split(",")

        wneSubject = wet.get_wne_subject(subjectName)
        wneProject = wet.get_wne_project(args["--projectName"])

        if args["--exclusionsPath"]:
            assert args["--exclusionsPath"].endswith("tsv")
            assert Path(args["--exclusionsPath"]).exists()
            print(args["--exclusionsPath"])
            exclusions = pd.read_csv(args["--exclusionsPath"], sep="\t")
        else:
            exclusions = None

        sorting_pipeline = SpikeInterfaceSortingPipeline(
            wneProject,
            wneSubject,
            args["--experimentName"],
            args["--aliasName"],
            probe,
            basename=args["--basename"],
            rerun_existing=args["--rerun_existing"],
            n_jobs=args["--n_jobs"],
            options_source=args["--options_source"],
            exclusions=exclusions,
        )
        # Load raw recording and semgments
        sorting_pipeline.get_raw_si_recording()
        print(f"Sorting pipeline: {sorting_pipeline}\n")
        print(f"Pipeline opts: {sorting_pipeline.opts}")
        print(f"Raw recording:\n {sorting_pipeline._raw_si_recording}")

        if args["--prepro_only"]:
            print("--prepro_only==True: Run only preprocessing")
            sorting_pipeline.run_preprocessing()
        else:
            print("Run full pipeline")
            sorting_pipeline.run_preprocessing()
            sorting_pipeline.run_sorting()

        print(f"Sorting pipeline: {sorting_pipeline}\n")
        print(f"Pipeline opts: {sorting_pipeline._opts}\n")
        print(f"Raw recording:\n {sorting_pipeline._raw_si_recording}\n")

        print(f"\n\n ...Done!")
