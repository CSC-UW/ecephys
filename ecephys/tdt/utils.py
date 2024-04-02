from probeinterface import Probe
import spikeinterface as si
import numpy as np
from pathlib import Path

try:
    import tdt

    HAS_TDT_PACKAGE = True
except ImportError:
    HAS_TDT_PACKAGE = False


def read_and_save_tdt_recording_as_bin(
    tdt_block_path: Path, store: str, t1: float, t2: float, output_path: Path, dtype="float32"
) -> None:
    assert HAS_TDT_PACKAGE

    blk = tdt.read_block(
        tdt_block_path, store=store, evtype=["streams"], t1=t1, t2=t2
    )  # assumes the use of all channels (16), all channels have same sampling rate and same number of samples.
    print(f"Loading block data at {tdt_block_path}, t1={t1}, t2={t2}")
    data = blk.streams[store].data  # ndarray of shape (n_channels, n_samples)
    print(f"Saving {data.shape}-array to {output_path}")
    output_path.parent.mkdir(exist_ok=True, parents=True)
    data.astype(dtype).tofile(f"{output_path}")
    print(f"Done")


def set_probe_and_locations(si_recording: si.BaseRecording, probe_spacing: float = None):
    DF_PROBE_SPACING = 50
    if probe_spacing is None:
        probe_spacing = DF_PROBE_SPACING
    assert probe_spacing == DF_PROBE_SPACING # Double check otherwise

    CONTACT_RADIUS = 7.5
    POLYGON = [(-10, 50), (0, 0), (10, 50), (40, 850), (-40, 850)]

    # create the probe object and set assign it to the concatenated recording
    nchans = len(si_recording.get_channel_ids())
    assert nchans == 16

    # Contacts
    positions = np.zeros((nchans, 2))
    for i in range(nchans):
        x = 0
        y = (
            nchans - (i + 1)
        ) * probe_spacing + probe_spacing  # Invert here because channel 1 is most superficial and I'm getting lost with prb.set_contacts_ids() etc and don't want to invert stuff
        positions[i] = x, y

    prb = Probe(ndim=2, si_units="um")
    prb.set_contacts(positions=positions, shapes="circle", shape_params={"radius": CONTACT_RADIUS})

    # Geometry
    prb.set_planar_contour(POLYGON)
    prb.set_device_channel_indices(np.arange(nchans))

    # Assign contact ids otherwise forgetten when Recoding.set_probe
    # NNXr-1 is most superficaia of the probe
    # chan_ids = si_recording.get_channel_ids()
    # prb.set_contact_ids(chan_ids)

    si_recording.set_probe(prb, in_place=True)

    return si_recording
