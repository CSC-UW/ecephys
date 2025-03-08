import json
from pathlib import Path

import dask
import dask.array as da
import dask_image as di
import numpy as np
import pandas as pd
import scipy
import skimage
import spikeinterface.preprocessing as SI_pre
import xarray as xr
import zarr
from dask.distributed import Client, LocalCluster

import ecephys.utils
import wisc_ecephys_tools as wet

"""
Object Oriented Implementation of an image segmentation based method for
off detection developed by Graham Findlay and Tom Bugnon. 

Code written by Jack Walter (jack.walter122@gmail.com), 10/23"""


class ImsegOffDetection:
    def __init__(
        self,
        subject,
        probe,
        experiment,
        project,
        time_interval,
        alias="full",
        start_channel=None,
        end_channel=None,
        is_preprocessed=False,
        has_thresholds=False,
        save_thresholds=False,
        chunk_size="5s",
    ) -> None:
        self.subject = subject
        self.probe = probe
        self.experiment = experiment
        self.project = project
        self.alias = alias
        self.time_interval = time_interval

        if start_channel == None:
            self.start_channel = 0
        else:
            self.start_channel = start_channel

        if end_channel == None:
            self.end_channel = 384
        else:
            self.end_channel = end_channel

        self.is_preprocessed = is_preprocessed
        self.save_thresholds = save_thresholds
        self.has_thresholds = has_thresholds

        if not type(chunk_size) == str:
            raise TypeError(
                "chunk_size must be a string of form 'Xs' where X is the duration of the chunk"
            )
        else:
            self.chunk_size = chunk_size
        self.wne_project = wet.get_wne_project("off_det_analysis")
        self.project_dir = self.wne_project.dir
        self.outdir = Path(
            self.project_dir
            / self.subject
            / self.probe
            / f"{self.time_interval[0]}_{self.time_interval[1]}"
        )

        self.si_recording, self.segments = wet.get_sglx_subject(
            self.subject
        ).get_si_recording(
            self.experiment,
            alias=self.alias,
            stream="ap",
            probe=self.probe,
            sampling_frequency_max_diff=1e-6,
        )
        self.probe_x_corrds = None
        self.probe_y_coords = None
        self.chan_perm = None

        self.get_chan_idx_id_perm()

        self.sglx_project = wet.get_sglx_project(self.project)
        self.sample2time = self.sglx_project.get_sample2time(
            self.subject, self.experiment, alias, self.probe, "sorting"
        )

    """
    Run image segmentation pipeline. General steps are as follows
        -Preprocess raw data using WISC SI preprocessing pipeline
        -Establish per channel thresholds for 'off' defined as 
            1 standard deviation below the mean value for the channel
            over the duration of the time interval (NOTE: A significant 
            period of NREM should be used to establish these thresholds.
            This program has been tested with 2 hours, however it may be
            possible to use less. )
        -Apply gaussian blur to preprocessed traces and perform threshold based
        segmentation using previously established thresholds 
    """

    def run(self):
        # check if needed directories exist
        if self.project_dir / self.subject not in Path.iterdir(self.project_dir):
            Path.mkdir(self.project_dir / self.subject)

        if self.project_dir / self.subject / self.probe not in Path.iterdir(
            self.project_dir / self.subject
        ):
            Path.mkdir(self.project_dir / self.subject / self.probe)

        print(f"Saving run in {self.outdir}")
        if self.outdir not in Path.iterdir(
            self.project_dir / self.subject / self.probe
        ):
            Path.mkdir(self.outdir)
        # Timestampts and interval samples are saved to be reused,
        frame_timestamps_fname = (
            self.project_dir / self.subject / self.probe / "timestamps.npy"
        )
        interval_samples_fname = self.outdir / "interval_samples.json"

        frame_timestamps, interval_sample_ids = self.get_timestamps_interval_samples(
            frame_timestamps_fname,
            interval_samples_fname,
        )

        start_frame = interval_sample_ids[str(self.time_interval)][0]
        end_frame = interval_sample_ids[str(self.time_interval)][1]
        # Get timestamps that correspond to samples in time_interval
        slice_timestamps = frame_timestamps[start_frame:end_frame]
        """
        Per Tom this may need to be changed to whiten properly
        ie. preprocess full recording and then do frame slice
        """
        self.si_recording = self.si_recording.frame_slice(
            start_frame=start_frame, end_frame=end_frame
        )
        print(self.si_recording)
        if not self.is_preprocessed:
            self.si_recording = self.preprocess_si(self.start_channel, self.end_channel)
            print(self.si_recording)
            self.save_preprocessed_recording(self.si_recording)

        # Read preprocessed zarr store from disk
        print("Reading from zarr store")
        traces_path = str(Path(self.outdir / "prepro.zarr/traces_seg0"))
        traces = zarr.open(store=traces_path, mode="r")
        traces = da.from_zarr(traces)

        traces_xr = self.wrap_xr(traces, slice_timestamps)
        print(traces_xr)
        # Just free up memory
        del frame_timestamps
        del interval_sample_ids

        lbl_tmp_store = self.get_offs(traces_xr)
        offs, off_boundaries = self.make_dfs(traces_xr, lbl_tmp_store)

        offs = offs.sort_values(by=["start_time"])
        shared = wet.get_wne_project("shared")
        hg = shared.load_float_hypnogram(self.experiment, self.subject)
        # Uses start time of each off period to determine circadian states
        off_starts = offs["start_time"].to_numpy()
        states = hg.get_states(off_starts)
        offs["state"] = states

        # Save offs and boundaries
        t = f"{self.time_interval[0]}_{self.time_interval[1]}"
        offs_fname = f"{t}.offs.htsv"
        boundaries_fname = f"{t}.bound.npy"
        ecephys.utils.write_htsv(offs, self.outdir / offs_fname)
        np.save(arr=off_boundaries, file=self.outdir / boundaries_fname)
        print("Yay done!")
        return

    """
    Helper method to check if a file exists
    """

    def has_file(self, fname: Path, outdir: Path):
        if fname in Path.iterdir(outdir):
            return True
        else:
            return False

    def _get_frame_timestamps(self, segments, frame_timestamps_fname: Path, dir: Path):
        # IF timestamps exist on disk , load
        if self.has_file(frame_timestamps_fname, dir):
            print("Loading frame_timestamps")
            return np.load(frame_timestamps_fname)

        # Else construct timestamps using sample to time and save to disk
        else:
            print("Creating frame timestamps file")
            total_n_samples = segments.nSegmentSamp.sum()
            frame_ids = da.arange(0, total_n_samples, 1).compute()
            frame_timestamps = self.sample2time(frame_ids)
            # Write timestamps
            print("Writing timestamps file")
            np.save(arr=frame_timestamps, file=frame_timestamps_fname)
            return frame_timestamps

    """
    Compute the recording sample ids and sample2time, computes the timestamps that occur in time_interval
    """

    def get_timestamps_interval_samples(
        self, frame_timestamps_fname, interval_samples_fname
    ):
        if self.has_file(interval_samples_fname, self.outdir):
            print("Loading interval_samples file")
            # Either loads file containing recording timestamps from disk, or uses sampe2time to compute them
            # and then saves to file
            frame_timestamps = self._get_frame_timestamps(
                self.segments,
                frame_timestamps_fname,
                self.project_dir / self.subject / self.probe,
            )
            with open(interval_samples_fname, "r") as f:
                interval_samples = json.load(f)
            return frame_timestamps, interval_samples

            # Load json
            # The frame tim
        else:
            frame_timestamps = self._get_frame_timestamps(
                self.segments,
                frame_timestamps_fname,
                self.project_dir / self.subject / self.probe,
            )
            interval_samples = {str(self.time_interval): []}
            # If timestamp is within interval, add corresponding frame index to samples list
            # TODO: This process can (and should) be sped up by approximating the index of the first required
            #      timestamp (ie fs * start time) rather than traversing the whole list of timestamps.
            #      One would need to account for exmptPrb
            start_idx = None
            end_idx = None
            for i in range(len(frame_timestamps)):
                if (
                    frame_timestamps[i] >= self.time_interval[0]
                    and frame_timestamps[i] <= self.time_interval[1]
                ):
                    if start_idx is None:
                        start_idx = i
                # all timestamps have been found
                if frame_timestamps[i] > self.time_interval[1]:
                    break
            end_idx = i - 1
            first_last_sample = [start_idx, end_idx]
            print(first_last_sample)
            print(frame_timestamps[start_idx], frame_timestamps[end_idx])

            interval_samples[str(self.time_interval)] = first_last_sample

            print("Writing interval samples file")
            # TODO: Storing the first and last sample as a json is not necessary, could be refactored to .npy
            with open(interval_samples_fname, "w") as f:
                json.dump(interval_samples, f)

            return frame_timestamps, interval_samples

    """
    Apply WISC spike sorting preprocesing pipeline steps
        Args:
                si_recording (SpikeInterface.BaseRecording): recording to be preprocessed
                start_channel (int): first channel in range of interest
                end_channel (int): end channel in range of interest
                
        Return: 
                rec: The preprocessed recording
    """

    def preprocess_si(self, start_channel, end_channel):
        # TODO: This method for channel selection needs to be revisted
        #      and changed to select for channels within a given depth
        ids = self.si_recording.get_channel_ids()
        req_channels = []
        for id in ids:
            idx = int(id.split("AP", 1)[1])
            if idx >= start_channel and idx < end_channel:
                req_channels.append(id)

        rec = self.si_recording.channel_slice(req_channels)

        rec = SI_pre.bandpass_filter(rec, 300, 12000)
        rec = SI_pre.phase_shift(rec)
        # Detect bad channels for interpolation
        bad_channels = SI_pre.detect_bad_channels(rec)
        rec = SI_pre.interpolate_bad_channels(rec, bad_channels[0])
        rec = SI_pre.common_reference(rec, reference="local", operator="median")
        # Whitening needed to enforce equal variance for gaussian blur
        rec = SI_pre.whiten(rec, dtype="float32")
        rec = SI_pre.rectify(rec)
        return rec

    """
    Save and si_recording to a zarr store
        Args:
                si_recording (SpikeInterface.BaseRecording)
    """

    def save_preprocessed_recording(self, recording):
        # job args can be changed baed on memory needs
        job_kwargs = dict(n_jobs=0.5, chunk_duration=self.chunk_size, progress_bar=True)
        if Path(self.outdir / "prepro.zarr").exists():
            import shutil

            shutil.rmtree(Path(self.outdir / "prepro.zarr"))
        recording.save_to_zarr(
            folder=Path(self.outdir / "prepro"), format="zarr", **job_kwargs
        )

    """
    Wraps an array of traces in XArray.DataArray
        Args:
            traces(array-like): recording traces
            times(list): list of timestamps corresponding to samples in traces

        Return:
            a: DataArray wrapping of traces sorted by channel depth
            with timestamps and channel coordinates. Dimensions are (channel, time)
    """

    def wrap_xr(self, traces, times):
        fs = self.si_recording.get_sampling_frequency()
        # Check for time on x axis, if not transpose
        d = traces.shape
        if d[0] > d[1]:
            traces = traces.T

        if traces.shape[0] != len(self.chan_perm):
            raise Exception(
                "No. of channels in traces array must be equal to chan_perm"
            )

        traces = traces[self.chan_perm, :]
        # TODO This should be changed from using a range slice to a list of included channels
        channels = self.si_recording.get_channel_ids()
        channels = channels[self.start_channel : self.end_channel]
        channels = channels[self.chan_perm]

        a = xr.DataArray(
            data=traces,
            dims=("channel", "time"),
            coords={
                "time": times,
                "channel": channels,
                "x": ("channel", self.probe_x_coords.flatten()),
                "y": ("channel", self.probe_y_coords.flatten()),
            },
            attrs={"units": "unscaled", "fs": fs},
            name="traces",
        )
        return a

    """
    Use channel depths to infer correct sptial ordering of channel indices 
    """

    def get_chan_idx_id_perm(self):
        x_coords = self.si_recording.get_channel_locations(axes="x")
        y_coords = self.si_recording.get_channel_locations(axes="y")
        # Infer correct order of channel indices based on channel depth
        self.chan_perm = y_coords.argsort(axis=0).flatten()
        # Reorder x and y coords according to channel depth
        x_coords = x_coords[self.chan_perm, :]
        y_coords = y_coords[self.chan_perm, :]
        self.probe_x_coords = x_coords
        self.probe_y_coords = y_coords

    """
    Apply image segmentation algorithm to detech off periods 
    
    Intermediate steps are saved to disk as zarr stores to allow lazy processing 
    of successive steps
    
        Args:
                traces_xr (XArray.DataArray): recording traces wrapped with xarray,
                                              should have dim = (N-channels, time)
                                              
                thresholds (array-like or None): Vector containing per channel thresholds
                                                 if None it will be computed from the data.
                                                 thresholds.shape = (N-channels,)
                                                 
        Return:
                lbl_tmp_store (Pathlike): Path to zarr store containing labeled features
    """

    def get_offs(self, traces_xr, thresholds=None):
        # Set relevant dask configs
        dask.config.set({"array.slicing.split_large_chunks": False})
        dask.config.set({"distributed.admin.tick.limit": "180s"})
        dask.config.set(
            {"distributed.nanny.pre_spawn_environ.MALLOC_TRIM_THRESHOLD_": 0}
        )
        # Start up dask client
        cluster = LocalCluster(n_workers=10, memory_limit="60GB")
        client = Client(cluster)
        # Apply gaussian blur and save result to disk
        blur = traces_xr.copy()
        blur.data = di.ndfilters.gaussian_filter(blur.data, sigma=(5, 300))
        # Save blured data to zarr store, may be helpful to rechunk blur before writing
        blur_tmp_store = Path(self.outdir / "blur.tmp.zarr")
        da.to_zarr(blur.data.rechunk(), url=blur_tmp_store, overwrite=True)

        blur = da.from_zarr(blur_tmp_store)

        if thresholds == None:
            if self.has_thresholds:
                thresholds_store = Path(
                    self.project_dir / self.subject / self.probe / "thresholds.zarr"
                )
                thresholds = da.from_zarr(thresholds_store)
            else:
                client.close()
                thresholds = self.get_thresholds(blur)
                cluster = LocalCluster(n_workers=10, memory_limit="60GB")
                client = Client(cluster)

        # Find off features using thresholds
        off = da.where(blur.T < thresholds, True, False)
        lbl_img, n_lbls = di.ndmeasure.label(off)
        lbl_tmp_store = Path(self.outdir / "lbl.tmp.zarr")
        da.to_zarr(lbl_img, url=lbl_tmp_store, overwrite=True)

        client.close()

        return lbl_tmp_store

    """
    Use preprocessed traces to calculate per channel thresholds for segmentation.
    Thresholds are 1 standad deviation below the mean value of the channel
    
    Args:
            traces (dask.array.Array): Dask array representation of traces, 
                                       SHOULD HAVE GUASSIAN BLUR APPLIED 
            
    Return:
            thresholds (array-like) Threshold vector
    
    
    """

    def get_thresholds(self, traces):
        # Start up dask client
        # Client args should be changed based on amount of memory you're willing to use
        cluster = LocalCluster(n_workers=10, memory_limit="60GB")
        client = Client(cluster)

        log = da.log(traces)
        # Save to tmp store to allow lazy processing of next steps
        da.to_zarr(log, url=Path(self.outdir / "log.tmp.zarr"), overwrite=True)
        log = da.from_zarr(Path(self.outdir / "log.tmp.zarr"))

        u = da.mean(log, axis=1)
        std = da.std(log, axis=1)
        channel_thresholds = da.exp(u - std)
        # Option to save thresholds for use in future runs
        if self.save_thresholds:
            da.to_zarr(
                channel_thresholds,
                url=Path(
                    self.project_dir / self.subject / self.probe / "thresholds.zarr"
                ),
                overwrite=True,
            )

        channel_thresholds = channel_thresholds.compute()

        client.close()
        return channel_thresholds

    def _get_strict_n_frames(self, row_indices, col_indicies):
        rc = pd.DataFrame({"row": row_indices, "col": col_indicies})
        return rc.groupby("col").count()["row"].max()

    """
    Using the image features returned by get_offs(), create a DataFrame to store statistics
    about and find boundary of each off period
    
    Args:
            data (XArray.DataArray): DataArray containing coordiate and timestamp info about traces
                                     used for off detection
            lbl_tmp_store(Pathlike): path to zarr store containing labeled features generated by get_offs()
            
    Return:
            offs (Pandas.DataFrame): DataFrame containing information about each off (duration, area, start and end time, etc)
            off boundaries (numpy array-like): x and y coordinates for boundaries of each off period 

    """

    def make_dfs(self, data, lbl_tmp_store):
        cluster = LocalCluster(n_workers=10, memory_limit="60GB")
        client = Client(cluster)

        lbl_img = da.from_zarr(lbl_tmp_store)

        result_dtypes = {
            "area": np.dtype("int64"),
            "start_frame": np.dtype("int64"),
            "end_frame": np.dtype("int64"),
            "strict_n_frames": np.dtype("int64"),
            "min_chan_ix": np.dtype("int64"),
            "max_chan_ix": np.dtype("int64"),
            "label": np.dtype("int32"),
        }
        print("got dtypes")
        print(result_dtypes)

        lbl_img = lbl_img.compute()
        print("reached")
        # This step is very memory intensive but necessary to determine off duration/area
        # dask does not currently have an equivalent function for value_indices but it should be used if ever implemented
        val_ixs = scipy.ndimage.value_indices(
            lbl_img, ignore_value=0
        )  # {lbl: (row/time_indices, col/chan_indices)}
        del lbl_img

        print("reached")
        _lbls = np.sort(list(val_ixs.keys()))
        areas = pd.DataFrame(
            [val_ixs[lbl][0].size for lbl in _lbls], columns=["area"], index=_lbls
        )
        start_frames = pd.DataFrame(
            [val_ixs[lbl][0].min() for lbl in _lbls],
            columns=["start_frame"],
            index=_lbls,
        )
        end_frames = pd.DataFrame(
            [val_ixs[lbl][0].max() for lbl in _lbls], columns=["end_frame"], index=_lbls
        )
        strict_n_frames = pd.DataFrame(
            [self._get_strict_n_frames(*val_ixs[lbl]) for lbl in _lbls],
            columns=["strict_n_frames"],
            index=_lbls,
        )
        min_chan_ixs = pd.DataFrame(
            [val_ixs[lbl][1].min() for lbl in _lbls],
            columns=["min_chan_ix"],
            index=_lbls,
        )
        max_chan_ixs = pd.DataFrame(
            [val_ixs[lbl][1].max() for lbl in _lbls],
            columns=["max_chan_ix"],
            index=_lbls,
        )
        del val_ixs

        # df = pd.concat([cm_coords, pk_values, areas, start_frames, end_frames, strict_n_frames, min_chan_ixs, max_chan_ixs], axis=1).dropna()
        df = pd.concat(
            [
                areas,
                start_frames,
                end_frames,
                strict_n_frames,
                min_chan_ixs,
                max_chan_ixs,
            ],
            axis=1,
        ).dropna()
        df["label"] = df.index
        df = df.astype(result_dtypes)

        y = data.y.values
        t = data.time.values

        df["start_time"] = t[df["start_frame"].values]
        df["end_time"] = t[df["end_frame"].values]
        df["duration"] = df["end_time"] - df["start_time"]
        df["strict_duration"] = df["strict_n_frames"].values / data.fs
        df["min_y"] = y[df["min_chan_ix"].values]
        df["max_y"] = y[df["max_chan_ix"].values]
        df["yspan"] = df["max_y"] - df["min_y"]
        # Excludes all offs with a min duration of 60ms and y span of 400um, these params can be changed
        # to increase/decrease sensitivity
        offs = df.loc[(df["yspan"] >= 400) & (df["strict_duration"] >= 0.060)].copy()
        del df

        print("created offs df")

        lbl_img = da.from_zarr(lbl_tmp_store)
        mask = da.isin(lbl_img, offs["label"].values)
        mask_tmp_store = Path(self.outdir / "mask.tmp.zarr")
        da.to_zarr(mask, url=mask_tmp_store, overwrite=True)

        mask = da.from_zarr(mask_tmp_store)
        off_img = da.where(mask, lbl_img, 0)
        off_tmp_store = Path(self.outdir / "off.tmp.zarr")
        da.to_zarr(off_img, url=off_tmp_store, overwrite=True)
        del lbl_img
        del mask
        # Get the boundaries of each feature
        off_img = da.from_zarr(url=off_tmp_store)
        sk_boundaries = da.from_array(
            skimage.util.apply_parallel(
                skimage.segmentation.find_boundaries, off_img, compute=True
            )
        )
        off_boundaries = da.where(sk_boundaries)

        print("done")
        x = off_boundaries[0].compute()
        y = off_boundaries[1].compute()
        off_boundaries = [x, y]

        client.close()
        return offs, off_boundaries
