# SGLXarray
Simple tools for loading [SpikeGLX](https://billkarsh.github.io/SpikeGLX/) data, including useful metadata, as [xarray](https://docs.xarray.dev/en/stable/) objects.  
In use at the [Center for Sleep and Consciousness (CSC-UW)](https://centerforsleepandconsciousness.psychiatry.wisc.edu/research-overview/#SLEEP-target-element).  
Tested with Neuropixel 1.0 probes only.

## Basic usage

### Loading binary data
![QuickStart](https://user-images.githubusercontent.com/4753005/159067426-b5818765-7b11-414e-8f01-f3126a899376.png)  
See [example usage](example.ipynb) for more details.  
I do have functions for loading digital channels recorded with Multifunction I/O cards, let me know if you need me to integrate them urgently.  

### ImecMap objects
IMRO and CMP are parsed from the binary file's metadata by default and loaded as an attribute in the returned xarray object. There are also other utilities for loading these tables without binary data.   
![ImecMap](https://user-images.githubusercontent.com/4753005/159067440-3f3357f0-f2fb-4de4-b735-434511754484.png)  
See [example usage](example.ipynb) for more details. 
