import numpy as np

from scipy.signal import butter, filtfilt, medfilt

from . import rms


def get_good_channels(
    raw_data_file,
    num_channels,
    sample_rate,
    bit_volts,
    noise_threshold=20,
    ignored_channels=None,
    channel_map=None,
    ap_lf_chans=None,
):
    """Compute noise channels for a data file. Return mask of good channels.

        - 3rd order band-pass butter filter at 10Hz-10000Hz
        - rms of filtered signal per channel (-> nchan x 1 array)
        - 11-kernel median filtering across channels
        - bad if rms of a channel is <noise_threshold> above median of
          surrounding channels

    Args:
        raw_data_file (path-like): SGLX .bin file containing (nchan x
            ntimepoints) int16 values
        num_channels, sample_rate: For SGLX .bin file
        bit_volts: Conversion factor for raw int16 values

    Kwargs:
        noise_threshold (float): Difference of RMS to median of neighboring
            channels to consider a channel as bad. Lower is more
            conservative (default 20)
        ignored_channels (list(int)): 0-indexed list of channels excluded from
            analysis and marked as noise_channels (eg [191])
        ap_lf_chans (array-like): Indices of data channels (used to exclude
            SYNC)
        channel_map (array-like): Mapping used to reorder channels by physical
            coordinate

    Return:
        (nchans,)-array of GOOD channels.
    """
    if ignored_channels is None:
        ignored_channels = []
    if ap_lf_chans is None:
        ap_lf_chans = np.array(range(num_channels))
    if channel_map is None:
        channel_map = np.array(range(len(ap_lf_chans)))
    assert all([i in ap_lf_chans for i in ignored_channels])

    noise_delay = 5  # in seconds
    noise_interval = 10  # in seconds

    raw_data = np.memmap(raw_data_file, dtype="int16")

    num_samples = int(raw_data.size / num_channels)

    data = np.reshape(raw_data, (num_samples, num_channels))

    start_index = int(noise_delay * sample_rate)
    end_index = int((noise_delay + noise_interval) * sample_rate)

    if end_index > num_samples:
        print("noise interval larger than total number of samples")
        end_index = num_samples

    b, a = butter(3, [10 / (sample_rate / 2), 10000 / (sample_rate / 2)], btype="band")

    D = data[start_index:end_index, :] * bit_volts

    D_filt = np.zeros(D.shape)

    for i in range(D.shape[1]):
        D_filt[:, i] = filtfilt(b, a, D[:, i])

    rms_values = np.apply_along_axis(rms, axis=0, arr=D_filt)

    # Use only ap/lf channels
    rms_values_aplf = rms_values[ap_lf_chans]
    assert len(ap_lf_chans) == len(channel_map)

    # Reorder rms values according to channel map
    # We detect bad channels on reordered data, and revert back to original
    # (file) indices at the end
    rms_aplf_sorted = rms_values_aplf[channel_map]
    ignored_channels_sorted = channel_map[ignored_channels]

    # Find noise channels amongst non-ignored chans in sorted dataset
    keep_channels_sorted = np.array(
        [i for i in range(len(rms_aplf_sorted))
         if i not in ignored_channels_sorted]
    )
    # Find rms values too far above surrounding values
    # Use only non-ignored channels
    rms_values_masked = rms_aplf_sorted[keep_channels_sorted]
    medfilt_masked = medfilt(rms_values_masked, 11)
    above_median_masked = rms_values_masked - medfilt_masked
    noise_chan_masked = above_median_masked > noise_threshold

    print(f"N ignored channels: {len(ignored_channels)} - ", end="")
    print(f"N detected noise channels: {sum(noise_chan_masked)}")

    # broadcast into original shape (nchan,)
    good_chans_aplf_sorted = np.ones((len(ap_lf_chans),), dtype=bool)
    good_chans_aplf_sorted[ignored_channels_sorted] = False  # originally ignored chans are masked
    good_chans_aplf_sorted[
        keep_channels_sorted[np.where(noise_chan_masked)[0]]
    ] = False  # detected bad channels are masked

    # Revert the sort
    arg_chanmap = np.argsort(channel_map)
    bad_idx = arg_chanmap[np.where(~good_chans_aplf_sorted)[0]]
    good_chans = np.ones((len(ap_lf_chans),), dtype=bool)
    good_chans[bad_idx] = False
    return good_chans
