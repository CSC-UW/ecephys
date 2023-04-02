#!/usr/bin/env python
# coding: utf-8

"""Run postprocessing pipeline.

Modify script to change default argument values

Usage:
  run_postprocessing_pipeline.py [options] [--input <subjectName>,<probeName>,<sorting_basename>]...

Options:
  -h --help                          Show this screen.
  --input==<subjectName,probeName,sortingBasename>   (Repeatable) Comma-separated pair of the form `<subjectName>,<probeName>,<sorting_basename>`
  --rerun_existing                   Rerun rather than load things.
  --options_source==<ofn>            Source of options file (applied to all input datasets) [default: {OPTIONS_SOURCE}]
  --projectName==<pn>                Project name [default: {PROJECT_NAME}]
  --hypnogramProjectName==<pn>       Where to pull hypnograms from
  --experimentName==<en>             Exp name [default: {EXPERIMENT_NAME}]
  --aliasName==<an>                  Alias name [default: {ALIAS_NAME}]
  --postprocessingName==<ppn>        Name of postprocessing output dir. [default: {POSTPROCESSING_NAME}]
  --n_jobs=<n_jobs>                  Number of jobs for all spikeinterface functions. [default: {N_JOBS}]

"""
from docopt import docopt

import wisc_ecephys_tools as wet
from ecephys import wne
from ecephys.wne.sglx.pipeline.postprocessing_pipeline import (
    SpikeInterfacePostprocessingPipeline,
)

subjectsDir = wet.get_subjects_directory()
projectsFile = wet.get_projects_file()

DEFAULT_VALUES = {
    "PROJECT_NAME": "my_project",
    "EXPERIMENT_NAME": "my_experiment",
    "ALIAS_NAME": "my_alias",
    "OPTIONS_SOURCE": "/path/to/opts/such/as/ecephys/wne/sglx/pipeline/template_postprocessing_opts.yaml",
    "POSTPROCESSING_NAME": "postprocessing_df",
    "N_JOBS": 10,
}

if __name__ == "__main__":

    args = docopt(__doc__.format(**DEFAULT_VALUES), version="Naval Fate 2.0")

    print(f"Running all subject/probe pairs: {args['--input']}")

    for subject_probe in args["--input"]:

        subjectName, probe, sorting_basename = subject_probe.split(",")

        subjLib = wne.sglx.SubjectLibrary(subjectsDir)
        projLib = wne.ProjectLibrary(projectsFile)
        wneSubject = subjLib.get_subject(subjectName)
        wneProject = projLib.get_project(args["--projectName"])

        if args["--hypnogramProjectName"]:
            hypnogram_source = projLib.get_project(args["--hypnogramProjectName"])
        else:
            hypnogram_source = None

        postpro_pipeline = SpikeInterfacePostprocessingPipeline(
            wneProject,
            wneSubject,
            args["--experimentName"],
            args["--aliasName"],
            probe,
            sorting_basename=sorting_basename,
            postprocessing_name=args["--postprocessingName"],
            rerun_existing=args["--rerun_existing"],
            n_jobs=args["--n_jobs"],
            options_source=args["--options_source"],
            hypnogram_source=hypnogram_source,
        )
        print(f"Postpro pipeline: {postpro_pipeline}\n")
        print(f"Pipeline opts: {postpro_pipeline._opts}")

        postpro_pipeline.run_postprocessing()

        print(f"\n\n ...Done!")
