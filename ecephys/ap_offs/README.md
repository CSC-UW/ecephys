# OFF detection from AP band data


### Example usage

```python
from ecephys import ap_offs
from dask.distributed import LocalCluster

raw_si_rec = ... # Raw AP band
raw_times = ... # Timestamps for raw AP band

# Hypnogram for thresholding
hg_thresholding = 

# Where we save the processed zarr recording
zarr_fpath = ...

# Process and downsample si rec
processing_opts = {
    "bandpass_filt_min": 100,
    "bandpass_filt_max": 12000,
    "gaussian_filt_max": 20,
    "decimation_factor": 100,
    "motion_correct": True,
}
detection_opts = {
    "median_filter_N_chans": 5,
    "median_filter_N_samples": 20,
    "std_threshold": 0.085,
    "std_threshold_ratio": 0.5,
    "mad_threshold": 1.5,
}

# You might want to use a different pipeline for non-neuropixels recording...
# (remove phase shift and CMR?)
pro_rec = ap_offs.preprocess.preprocess_neuropixels_si_recording(
    raw_si_rec, 
    times, # Real time for each sample
    opts=processing_opts,
    motion_npzfile=motion_npzfile,
)

# Save as zarr group
job_kwargs = dict(n_jobs=n_jobs, chunk_duration=f"10s", progress_bar=True)
rec.save(folder=zarr_fpath, overwrite=True, format="zarr", **job_kwargs)

# dask-based xarray with time, channel, y coordinates
da = ap_offs.utils.load_processed_zarr_as_xarray(fpath)

# Eg: Select region of interest
# da = da.sel(channel=...)

# multipro
cluster = LocalCluster(n_workers = n_jobs, memory_limit = '60GB')
client = cluster.get_client()

# Median filtering
da.data = dask_image.ndfilters.median_filter(
        da.data, 
        footprint=np.ones((detection_opts["median_filter_N_samples"], detection_opts["median_filter_N_chans"])),
    )

# Eg: select channels of interest based on STD
# NB: NEed even channel spacing
# da = da.sel(channel=...)

thresholds = ap_offs.detect.get_thresholds(da, detection_opts)
offs_df, lbl_ixs = ap_offs.detect.detect_ap_offs(da, thresholds, opts=detection_opts)

client.close()

# Plot
xlim = ...

f, ax = plt.subplots()
# Show processed ap trace
da.sel(time=slice(*xlim)).plot.imshow(x="time", y="y", ax=ax, cmap="inferno", vmin=0, vmax=1.5, interpolation=None)

# Add OFF mask
ap_offs.plot.add_ap_offs_overlay(
    da,  # The exact dataarray used for detection. Used to pull y/time coordinates
)




```

