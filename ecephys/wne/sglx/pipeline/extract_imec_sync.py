import logging
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
from tqdm.auto import tqdm

from ecephys import sync
from ecephys import utils
from ecephys.wne.sglx import SGLXProject
from ecephys.wne.sglx import SGLXSubject
from ecephys.wne.sglx import utils as wne_sglx_utils

logger = logging.getLogger(__name__)


def save_sglx_imec_barcodes(
    wne_project: SGLXProject, wne_subject: SGLXSubject, binfile: Path
):
    times, values = sync.get_sglx_imec_barcodes(binfile)
    barcodes = pd.DataFrame({"time": times, "value": values})
    [htsv_file] = wne_sglx_utils.get_sglx_file_counterparts(
        wne_project,
        wne_subject.name,
        [binfile],
        ".barcodes.htsv",  # TODO: Make a WNE constant
    )
    utils.write_htsv(barcodes, htsv_file)


def save_sglx_imec_ttls(
    wne_project: SGLXProject, wne_subject: SGLXSubject, binfile: Path
):
    rising, falling = sync.extract_ttl_edges_from_sglx_imec(binfile)
    ttls = pd.DataFrame({"rising": rising, "falling": falling})
    [htsvFile] = wne_sglx_utils.get_sglx_file_counterparts(
        wne_project,
        wne_subject.name,
        [binfile],
        ".ttls.htsv",  # TODO: Make a WNE constant
    )
    utils.write_htsv(ttls, htsvFile)


def do_probe(
    wne_project: SGLXProject,
    wne_subject: SGLXSubject,
    experiment: str,
    prb: str,
    fn: Callable,
    stream: str = "ap",
):
    ftab = wne_subject.get_experiment_frame(
        experiment, stream=stream, ftype="bin", probe=prb
    )
    for file in tqdm(list(ftab.itertuples()), desc="Files"):
        fn(wne_project, wne_subject, file.path)


def do_experiment(
    opts: dict,
    dest_project: SGLXProject,
    wne_subject: SGLXSubject,
    experiment: str,
    probes: Optional[list[str]] = None,
    stream: str = "ap",
):
    if opts["imSyncType"] == "square_pulse":
        fn = save_sglx_imec_ttls
    elif opts["imSyncType"] == "barcode":
        fn = save_sglx_imec_barcodes
    else:
        raise ValueError(f"Got unexpected value {opts['imSyncType']} for 'imSyncType'.")

    if probes is None:
        probes = wne_subject.get_experiment_probes(experiment)
    for probe in probes:
        do_probe(dest_project, wne_subject, experiment, probe, fn, stream=stream)
