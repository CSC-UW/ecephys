#!/usr/bin/env python
# coding: utf-8


from ecephys.wne.sglx.pipeline.sorting_pipeline import SortingPipeline, SpikeInterfaceSortingPipeline
import wisc_ecephys_tools as wet


subjectsDir = wet.get_subjects_directory()
projectsFile = wet.get_projects_file()

projectName = 'test_project'

subjectName = "CNPIX4-Doppio"
probe = 'imec1'

experimentName = 'novel_objects_deprivation'
aliasName = 'full'
opts_filepath = "/path/to/opts/such/as/ecephys/wne/sglx/pipeline/template_opts/template_pipeline_opts.yaml"

time_ranges = None

rerun_existing = True # Recompute drift estimates / rerun sorting


if __name__ == "__main__":

    sorting_pipeline = SpikeInterfaceSortingPipeline(
        subjectName,
        subjectsDir,
        projectName,
        projectsFile,
        experimentName,
        aliasName,
        probe,
        time_ranges=time_ranges,
        opts_filepath=opts_filepath,
        rerun_existing=rerun_existing,
    )
    sorting_pipeline

    print(f"Pipeline opts: {sorting_pipeline.opts}")
    print(f"Raw recording:\n {sorting_pipeline.raw_si_recording}")
    sorting_pipeline.run_preprocessing()