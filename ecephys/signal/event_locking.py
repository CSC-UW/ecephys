import pandas as pd
import numpy as np


def get_windows(event_times, window, trim='end'):
    """Return nevents x 2 array of windows around events, possibly trimmed."""
    assert trim in ['start', 'end']
    assert len(window) == 2
    assert window[0] < window[1]

    event_times = np.array(event_times)
    event_windows = np.transpose(
        np.array([event_times + window[0], event_times + window[1]])
    ) # nevents x 2

    # Trim overlapping windows
    nevents = len(event_times)
    for i in range(nevents):
        t_start = event_windows[i, 0]
        t_end = event_windows[i, 1]
        # End of window overlaps with next and we should trim the ends
        if (i < nevents - 1
            and t_end > event_windows[i+1, 0]
            and trim == 'end'):
            event_windows[i, 1] = event_windows[i+1, 0]
        # Start of window overlaps with previous and we should trim the starts
        if (i > 0
            and t_start < event_windows[i-1, 1]
            and trim == 'start'):
            event_windows[i, 0] = event_windows[i-1, 1]

    return event_windows


def get_locked_df(
    data_df,
    event_times,
    window,
    time_colname='time',
    trim='end',
    drop_out_of_window_datapoints=False,
):
    """Return timing relative to events to a frame.

    Parameters
    ----------
    data_df: pd.DataFrame, length (n_datapoints)
        Frame of data points. Must contain (absolute) times for data points
        information (time_colname kwarg)
    event_times: ndarray, shape (n_events,)
        List or array of event time used for time-locking.
    window: tuple
        `(t_start, t_end)` window used to align data times to events.
        Same unit as data_df[time_colname]
    time_colname: str, optional
        Name of column containing (absolute) timing information.  (default 'time')
    trim: str, optional
        'start' or 'end'. If windows around successive events overlap, do we
        trim the end of the earlier window or the start of the later one?
        (default 'end')
    drop_out_of_window_datapoints: bool, optional
        If true, return only datapoints within one of the event windows.
    
    Returns:
    data_df: pd.DataFrame, length (n_datapoints)
        Copy of original `data_df` frame, with the following columns added:
        `event_idx`: Idx of event we're locking the datapoint to
        `event_time`: Time of event we're locking the datapoint to
        `{time_colname}_rel`: Time relative to event
        `event_window_start`, `event_window_end`: Relative start and end of the window
            of the event the datapoint is assigned to. Vary depending on the
            trimming method.
    """
    event_windows = get_windows(event_times, window, trim=trim)

    time_rel_colname = f"{time_colname}_rel"

    data_df = data_df.copy()
    data_df[time_rel_colname] = None
    data_df['event_idx'] = None
    data_df['event_time'] = None
    data_df['event_window_start'] = None
    data_df['event_window_end'] = None

    for event_idx, event_time in enumerate(event_times):

        # Find indices of all datapoints within window
        event_window = event_windows[event_idx, :]
        event_rows = ((data_df[time_colname] >= event_window[0])
                      & (data_df[time_colname] < event_window[1]))

        # Add locked times
        data_df.loc[
            event_rows,
            time_rel_colname
        ] = data_df.loc[
            event_rows,
            time_colname
        ] - event_times[event_idx]

        # Add index of event and event window
        data_df.loc[event_rows, 'event_idx'] = event_idx
        data_df.loc[event_rows, 'event_time'] = event_time
        data_df.loc[event_rows, 'event_window_start'] = event_window[0]
        data_df.loc[event_rows, 'event_window_end'] = event_window[1]

    data_df['event_window_duration'] = data_df['event_window_end'] - data_df['event_window_start']

    if drop_out_of_window_datapoints:
        data_df = data_df.loc[
            ~pd.isna(data_df['event_idx'])
        ]

    return data_df


# TODO: Here we assumed that all the events appear
def get_event_firing_rates_df(
    locked_data_df,
    time_rel_colname='time_rel',
):
    """Return frame of cluster firing rates during event.

    Parameters
    ----------
    locked_data_df: pd.DataFrame, length (n_datapoints)
        Frame of data points, as returned by get_locked_df.
        Must contain (relative) times for data points and event information:
        The following columns are expected:
            `event_idx`, `event_time`, `{time_rel_colname}`, `event_window_duration`

    Returns:
    --------
    event_rates_df: pd.DataFrame, length (n_datapoints)
        Frame containing the firing rate within each event window for each cluster.
        Frame with ['cluster_id', 'event_idx'] MultiIndex and the folloeing columns:
            `event_num_spikes`, `event_time`, `event_window_duration`, `event_firing_rate`
    """

    assert all([c in locked_data_df for c in ['cluster_id', 't_rel', 'event_idx', 'event_time', 'event_window_duration']]), print(locked_data_df.columns)

    # TODO
    # Check all events appear in frame
    assert len(
        locked_data_df.event_idx.unique()
    ) == len(range(
        locked_data_df.event_idx.min(),
        locked_data_df.event_idx.max() + 1
    ))

    event_rates = locked_data_df.groupby(
        ['cluster_id', 'event_idx']
    ).agg(
        {
            't_rel': 'count',
            'event_time': 'first',
            'event_window_start': 'first',
            'event_window_end': 'first',
            'event_window_duration': 'first',
        }
    ).copy().rename(
        columns={'t_rel': 'event_num_spikes'}
    )
    event_rates['event_firing_rate'] = event_rates['event_num_spikes'] / event_rates['event_window_duration']

    return event_rates