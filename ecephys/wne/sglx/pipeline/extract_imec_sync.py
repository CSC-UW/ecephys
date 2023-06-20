import logging
from pathlib import Path

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


def do_experiment(
    dest_project: SGLXProject, wne_subject: SGLXSubject, experiment: str, **kwargs
):
    sessionIDs = wne_subject.get_experiment_session_ids(experiment)
    for id in sessionIDs:
        do_session(dest_project, wne_subject, id, **kwargs)


def do_session(
    dest_project: SGLXProject, wne_subject: SGLXSubject, session_id: str, **kwargs
):
    ftab = wne_subject.get_session_frame(session_id, ftype="bin", **kwargs)
    for binfile in tqdm(list(ftab.itertuples()), desc="Files"):
        if binfile.imSyncType in ["square_pulse", "random"]:
            save_sglx_imec_ttls(dest_project, wne_subject, binfile.path)
        elif binfile.imSyncType == "barcode":
            save_sglx_imec_barcodes(dest_project, wne_subject, binfile.path)
        else:
            raise ValueError(
                f"Got unexpected value {binfile.imSyncType} for 'imSyncType'."
            )
