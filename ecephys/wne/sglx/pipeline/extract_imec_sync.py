import logging
import pandas as pd
from ecephys import utils, sync
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def save_sglx_imec_barcodes(wneProject, wneSubject, binfile):
    times, values = sync.get_sglx_imec_barcodes(binfile)
    barcodes = pd.DataFrame({"time": times, "value": values})
    [htsvFile] = wneProject.get_sglx_counterparts(
        wneSubject.name, [binfile], ".barcodes.htsv"  # TODO: Make a WNE constant
    )
    utils.write_htsv(barcodes, htsvFile)


def save_sglx_imec_ttls(wneProject, wneSubject, binfile):
    rising, falling = sync.extract_ttl_edges_from_sglx_imec(binfile)
    ttls = pd.DataFrame({"rising": rising, "falling": falling})
    [htsvFile] = wneProject.get_sglx_counterparts(
        wneSubject.name, [binfile], ".ttls.htsv"  # TODO: Make a WNE constant
    )
    utils.write_htsv(ttls, htsvFile)


def do_probe(wneProject, wneSubject, experiment, prb, fn, stream="ap"):
    ftab = wneSubject.get_experiment_frame(
        experiment, stream=stream, ftype="bin", probe=prb
    )
    for file in tqdm(list(ftab.itertuples()), desc="Files"):
        fn(wneProject, wneSubject, file.path)


def do_experiment(opts, destProject, wneSubject, experiment, probes=None, stream="ap"):
    if opts["imSyncType"] == "square_pulse":
        fn = save_sglx_imec_ttls
    elif opts["imSyncType"] == "barcode":
        fn = save_sglx_imec_barcodes
    else:
        raise ValueError(f"Got unexpected value {opts['imSyncType']} for 'imSyncType'.")

    if probes is None:
        probes = wneSubject.get_experiment_probes(experiment)
    for probe in probes:
        do_probe(destProject, wneSubject, experiment, probe, fn, stream=stream)
