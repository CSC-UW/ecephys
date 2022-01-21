import os
from tkinter import W
import pandas as pd
from io import StringIO
from .external.SGLXMetaToCoords import XYCoord10

SUBPACKAGE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))


def get_file(filename):
    """Get a file from the ecephys/sglx/external/maps/ directory."""
    return os.path.join(SUBPACKAGE_DIRECTORY, "external", "maps", filename)


def load_cmp(fname):
    """Parse, check, and load a channel map file to a DataFrame.
    See SGLX documentation for explanations of DataFrame columns.

    Parameters
    ===========
    fname: string
        The name of the file to load, as it appears in the ecephys map directory.

    Examples:
    =========
    df = load_cmp('LongCol_1shank.imec.cmp')
    """
    assert fname.endswith(".imec.cmp"), "Unexpected filename."
    cmp_file = get_file(fname)
    with open(cmp_file, "r") as f:
        firstline = f.readline().rstrip()
    nAP, nLFP, nSY = (int(s) for s in firstline.split(","))
    cmp = pd.read_csv(
        cmp_file,
        sep=" |;",
        names=["label", "acq_order", "usr_order"],
        skiprows=1,
        engine="python",
    )
    cmp["stream"] = cmp["label"].str.extract("(\D+)")
    cmp["chan_id"] = cmp["label"].str.extract("(\d+)").astype(int)
    assert len(cmp) == (nAP + nLFP + nSY), "File header does not match content."
    return cmp


def load_imro(fname):
    """Parse, check, and load an IMRO file to a DataFrame.
    See SGLX documentation for explanations of DataFrame columns.

    Parameters
    ===========
    fname: string
        The name of the file to load, as it appears in the ecephys map directory.
    Examples:
    =========
    df = load_imro('LongCol_1shank.imro')
    """
    assert fname.endswith(".imro"), "Unexpected filename."
    imro_file = get_file(fname)
    with open(imro_file, "r") as f:
        contents = f.readline().rstrip()

    entries = contents.strip("()").split(")(")
    header_entry = entries[0]
    channel_entries = entries[1:]

    probe_type, n_chans = (int(s) for s in header_entry.split(","))
    assert probe_type == 0, "Only Neuropixel 1.0 probes are supported."
    imro = pd.read_csv(
        StringIO("\n".join(channel_entries)),
        delim_whitespace=True,
        names=["chan_id", "bank", "ref_id", "ap_gain", "lf_gain", "ap_highpass"],
    )
    imro["ele"] = imro.bank.values * n_chans + imro.chan_id.values
    imro["x"], imro["y"] = XYCoord10({}, imro.ele.values, False)
    return imro
