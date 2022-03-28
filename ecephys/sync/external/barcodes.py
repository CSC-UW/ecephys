import numpy as np

def extract_barcodes_from_times(
    on_times,
    off_times,
    inter_barcode_interval=10,
    bar_duration=0.03,
    barcode_duration_ceiling=2,
    nbits=32,
):
    """Read barcodes from timestamped rising and falling edges.

    Parameters
    ----------
    on_times : numpy.ndarray
        Timestamps of rising edges on the barcode line
    off_times : numpy.ndarray
        Timestamps of falling edges on the barcode line
    inter_barcode_interval : numeric, optional
        Minimun duration of time between barcodes.
    bar_duration : numeric, optional
        A value slightly shorter than the expected duration of each bar
    barcode_duration_ceiling : numeric, optional 
        The maximum duration of a single barcode
    nbits : int, optional
        The bit-depth of each barcode

    Returns
    -------
    barcode_start_times : list of numeric
        For each detected barcode, the time at which that barcode started
    barcodes : list of int
        For each detected barcode, the value of that barcode as an integer.

    Notes
    -----
    ignores first code in prod (ok, but not intended)
    ignores first on pulse (intended - this is needed to identify that a barcode is starting)

    """

    start_indices = np.diff(on_times)
    a = np.where(start_indices > inter_barcode_interval)[0]
    barcode_start_times = on_times[a + 1]

    barcodes = []

    for i, t in enumerate(barcode_start_times):

        oncode = on_times[
            np.where(
                np.logical_and(on_times > t, on_times < t + barcode_duration_ceiling)
            )[0]
        ]
        offcode = off_times[
            np.where(
                np.logical_and(off_times > t, off_times < t + barcode_duration_ceiling)
            )[0]
        ]

        currTime = offcode[0]

        bits = np.zeros((nbits,))

        for bit in range(0, nbits):

            nextOn = np.where(oncode > currTime)[0]
            nextOff = np.where(offcode > currTime)[0]

            if nextOn.size > 0:
                nextOn = oncode[nextOn[0]]
            else:
                nextOn = t + inter_barcode_interval

            if nextOff.size > 0:
                nextOff = offcode[nextOff[0]]
            else:
                nextOff = t + inter_barcode_interval

            if nextOn < nextOff:
                bits[bit] = 1

            currTime += bar_duration

        barcode = 0

        # least sig left
        for bit in range(0, nbits):
            barcode += bits[bit] * pow(2, bit)

        barcodes.append(barcode)

    return barcode_start_times, barcodes