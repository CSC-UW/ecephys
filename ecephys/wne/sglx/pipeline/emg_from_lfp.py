import logging
import ecephys as ece
from tqdm.auto import tqdm
from . import utils

logger = logging.getLogger(__name__)

DEFAULT_EMG_OPTIONS = dict(
    target_sf=20,
    window_size=25.0,
    wp=[300, 600],
    ws=[275, 625],
    gpass=1,
    gstop=60,
    ftype="butter",
)


def do_alias(wneProject, wneSubject, experiment, alias=None, **kwargs):
    opts = wneProject.load_experiment_subject_json(
        experiment, wneSubject.name, ece.wne.constants.EXP_PARAMS_FNAME
    )

    lfpTable = wneSubject.get_lfp_bin_table(experiment, alias, **kwargs)

    for lfpFile in tqdm(list(lfpTable.itertuples())):
        [emgFile] = wneProject.get_sglx_counterparts(
            wneSubject.name, [lfpFile.path], ece.wne.constants.EMG_EXT
        )
        sig = ece.sglxr.load_trigger(
            lfpFile.path,
            opts["probes"][lfpFile.probe]["emgFromLfpChans"],
            t0=lfpFile.tExperiment,
            dt0=lfpFile.dtExperiment,
        )
        emg = ece.xrsig.LFPs(sig).synthetic_emg(**DEFAULT_EMG_OPTIONS)
        ece.utils.save_xarray(emg, emgFile)

    for probe in lfpTable.probe.unique():
        utils.gather_and_save_alias_dataarray(
            wneProject,
            wneSubject,
            experiment,
            alias,
            probe,
            ece.wne.constants.EMG_EXT,
            ece.wne.constants.EMG_FNAME,
        )
