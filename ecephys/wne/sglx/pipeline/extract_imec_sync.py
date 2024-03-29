import logging
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from ecephys import sync
from ecephys import utils
from ecephys.wne import constants
from ecephys.wne.sglx import SGLXProject
from ecephys.wne.sglx import SGLXSubject
from ecephys.wne.sglx import utils as wne_sglx_utils

logger = logging.getLogger(__name__)


def extract_barcodes_from_saved_ttls(
    sglx_project: SGLXProject, sglx_subject: SGLXSubject, binfile: Path
):
    [ttl_file] = wne_sglx_utils.get_sglx_file_counterparts(
        sglx_project, sglx_subject.name, [binfile], constants.TTL_EXT
    )
    ttls = utils.read_htsv(ttl_file)
    times, values = sync.extract_barcodes_from_times(
        ttls["rising"].values, ttls["falling"].values, bar_duration=0.029
    )
    barcodes = pd.DataFrame({"time": times, "value": values})
    [barcode_file] = wne_sglx_utils.get_sglx_file_counterparts(
        sglx_project, sglx_subject.name, [binfile], constants.BARCODE_EXT
    )
    utils.write_htsv(barcodes, barcode_file)


def save_sglx_imec_ttls(
    wne_project: SGLXProject, sglx_subject: SGLXSubject, binfile: Path
):
    rising, falling = sync.extract_ttl_edges_from_sglx_imec(binfile)
    ttls = pd.DataFrame({"rising": rising, "falling": falling})
    [htsvFile] = wne_sglx_utils.get_sglx_file_counterparts(
        wne_project, sglx_subject.name, [binfile], constants.TTL_EXT
    )
    utils.write_htsv(ttls, htsvFile)


def do_experiment(
    experiment: str, sglx_subject: SGLXSubject, dest_project: SGLXProject, **kwargs
):
    sessionIDs = sglx_subject.get_experiment_session_ids(experiment)
    for id in sessionIDs:
        do_session(id, sglx_subject, dest_project, **kwargs)


def do_session(
    session_id: str, sglx_subject: SGLXSubject, dest_project: SGLXProject, **kwargs
):
    ftab = sglx_subject.get_session_frame(session_id, ftype="bin", **kwargs)
    for binfile in tqdm(list(ftab.itertuples()), desc="Files"):
        save_sglx_imec_ttls(dest_project, sglx_subject, binfile.path)
    for binfile in tqdm(list(ftab.itertuples()), desc="Files"):
        if binfile.imSyncType == "barcode":
            extract_barcodes_from_saved_ttls(dest_project, sglx_subject, binfile.path)
