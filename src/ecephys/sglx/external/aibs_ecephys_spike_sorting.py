import os
from pathlib import Path
from .readSGLX import readMeta

#########################################################################
#########################################################################
# Functions below are copied from Allen Institute's ecephys_spike_sorting
# package
#########################################################################
#########################################################################


def EphysParams(ap_band_file):
    # assume metadata file is in same directory as binary, Constuct metadata path

    # read metadata

    metaName, binExt = os.path.splitext(ap_band_file)
    metaFullPath = Path(metaName + ".meta")
    meta = readMeta(metaFullPath)

    if "imDatPrb_type" in meta:
        pType = meta["imDatPrb_type"]
        if pType == "0":
            probe_type = "NP1"
        else:
            probe_type = "NP" + pType
    else:
        probe_type = "3A"  # 3A probe

    sample_rate = float(meta["imSampRate"])

    num_channels = int(meta["nSavedChans"])

    uVPerBit = Chan0_uVPerBit(meta)

    return (probe_type, sample_rate, num_channels, uVPerBit)


# Return gain for imec channels.
# Index into these with the original (acquired) channel IDs.
#
def Chan0_uVPerBit(meta):
    # Returns uVPerBit conversion factor for channel 0
    # If all channels have the same gain (usually set that way for
    # 3A and NP1 probes; always true for NP2 probes), can use
    # this value for all channels.

    imroList = meta["imroTbl"].split(sep=")")
    # One entry for each channel plus header entry,
    # plus a final empty entry following the last ')'
    # channel zero is the 2nd element in the list

    if "imDatPrb_dock" in meta:
        # NP 2.0; APGain = 80 for all channels
        # voltage range = 1V
        # 14 bit ADC
        uVPerBit = (1e6) * (1.0 / 80) / pow(2, 14)
    else:
        # 3A, 3B1, 3B2 (NP 1.0)
        # voltage range = 1.2V
        # 10 bit ADC
        currList = imroList[1].split(sep=" ")  # 2nd element in list, skipping header
        APgain = float(currList[3])
        uVPerBit = (1e6) * (1.2 / APgain) / pow(2, 10)

    return uVPerBit