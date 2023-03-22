import numpy as np
import pandas as pd

from ecephys import sync
from ecephys import utils
from ecephys import wne


def get_sync_table(
    wneProject: wne.Project,
    wneSubject: wne.sglx.Subject,
    experiment: str,
    stream: str = "ap",
) -> pd.DataFrame:
    opts = wneProject.load_experiment_subject_json(
        experiment, wneSubject.name, wne.constants.EXP_PARAMS_FNAME
    )
    if opts["imSyncType"] != "barcode":
        raise NotImplementedError("Sync table generation only implemented for barcodes")

    syncStore = opts["tdt"]["barcode_store"]

    imec_barcode_times = []
    tdt_barcode_times = []
    barcode_values = []

    block_barcode_times, block_barcode_values = sync.get_tdt_barcodes(
        wneSubject.get_tdt_block_path(experiment), syncStore
    )
    ftab = wneSubject.get_experiment_frame(
        experiment, stream=stream, ftype="bin", probe="imec0"
    )
    for f in ftab.itertuples():
        [barcode_file] = wneProject.get_sglx_counterparts(
            wneSubject.name, [f.path], ".barcodes.htsv"
        )
        binfile_barcodes = utils.read_htsv(barcode_file)
        good_binfile_barcodes, binfile_slice, block_slice = sync.get_shared_sequence(
            binfile_barcodes["value"].values, block_barcode_values
        )
        barcode_values.append(good_binfile_barcodes)
        imec_barcode_times.append(
            binfile_barcodes["time"].values[binfile_slice] + f.expmtPrbAcqFirstTime
        )
        tdt_barcode_times.append(block_barcode_times[block_slice])

    barcode_values = np.concatenate(barcode_values)
    tdt_barcode_times = np.concatenate(tdt_barcode_times)
    imec_barcode_times = np.concatenate(imec_barcode_times)
    fit = sync.fit_times(
        tdt_barcode_times, imec_barcode_times, xname=syncStore, yname="imec0"
    )

    return pd.DataFrame(
        {
            "source": "tdt",
            "target": "imec0",
            "slope": [fit.coef_[0]],
            "intercept": [fit.intercept_],
        }
    )


def do_experiment(
    wneProject: wne.Project,
    wneSubject: wne.sglx.Subject,
    experiment: str,
    stream: str = "ap",
):
    sync_table = get_sync_table(wneProject, wneSubject, experiment, stream)

    f = wneProject.get_experiment_subject_file(
        experiment, wneSubject.name, f"tdt_sync.{stream}.htsv"
    )
    utils.write_htsv(sync_table, f)
