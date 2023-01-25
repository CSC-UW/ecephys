import logging
import neurodsp
import xarray as xr
from ecephys import sglxr, xrsig, utils, wne
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

DOWNSAMPLE_FACTOR = 4
CHUNK_SIZE = 2**16
CHUNK_OVERLAP = 2**10


def do_alias(
    opts,
    destProject,
    wneSubject,
    experiment,
    alias=None,
    **kwargs,
):
    """
    Parameters:
    -----------
    alias: str or None
        If None, do whole experiment


    IBL LFP pipeline
    -----
    1. 2-200Hz 3rd order butter bandpass, applied with filtfilt
    2. Destriping
        2.1 2 Hz 3rd order butter highpass, applied with filtfilt
        2.2 Dephasing
        2.3 Interpolation (inside brain)
        2.4 CAR
            2.4.1 Automatic gain control. What, exactly is this? Why do it?
            2.4.2 Median subtraction
    3. Decimation (10x)
        3.1 FIR anti-aliasing filter w/ phase correction

    WISC LFP pipeline
    -----
    1. Decimation (4x)
    2. Dephasing (adjusted for decimation)
    3. Interpolation

    Notes:
    -----
    - Time to load whole 2h file at once, all 384 channels: 6.5m
    - Time to load whole 2h file in 2**16 sample segments (~26s) with 2**10 sample overlap (~0.5s): ~15min
    - Time to dephase, interpolate, and decimate segments: 28m
    - Time to decimate, dephase, and interpolate: 7.5m.
        - Results are nearly identical. Resulting lfps are equivalent to within a picovolt.
        - Dephasing also continue to work just as well, provided your sample shifts account for the decimation.
    - Time to write float32 and float64 files are the same (~20s), but time to read float32 is twice as fast (5 vs 10s).
    - I have not yet tested whether using overlapping windows is truly necessary. It may not be, since the only filter here is the FIR antialiasing filter.
    - Note that the data here are NOT de-meaned.
    """
    lfpTable = wneSubject.get_lfp_bin_table(experiment, alias, **kwargs)
    for lfpFile in tqdm(list(lfpTable.itertuples())):
        [outFile] = destProject.get_sglx_counterparts(
            wneSubject.name, [lfpFile.path], ".nc"
        )
        outFile.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Loading {lfpFile.path.name}...")
        lfp = sglxr.load_trigger(
            lfpFile.path, t0=lfpFile.wneFileStartTime, dt0=lfpFile.wneFileStartDatetime
        )

        logger.info("Processing chunks...")
        wg = neurodsp.utils.WindowGenerator(
            ns=lfp.time.size, nswin=CHUNK_SIZE, overlap=CHUNK_OVERLAP
        )
        segments = list()
        for first, last in tqdm(list(wg.firstlast)):
            seg = lfp.isel(time=slice(first, last))
            seg = xrsig.NPX1LFPs(seg)
            seg = seg.decimate(q=DOWNSAMPLE_FACTOR)
            seg = seg.dephase()
            seg = seg.interpolate(opts["probes"][lfpFile.probe]["badChannels"])
            first_valid = 0 if first == 0 else int(wg.overlap / 2 / DOWNSAMPLE_FACTOR)
            last_valid = (
                seg.time.size
                if last == lfp.time.size
                else int(seg.time.size - wg.overlap / 2 / DOWNSAMPLE_FACTOR)
            )
            segments.append(seg.isel(time=slice(first_valid, last_valid)))

        logger.info("Concatenating chunks...")
        lfp = xr.concat(segments, dim="time")
        lfp.name = "lfp"

        logger.info(f"Saving to: {outFile}")
        utils.save_xarray(lfp, outFile, encoding={"lfp": {"dtype": "float32"}})

    logger.info("Done preprocessing LFPs!")
