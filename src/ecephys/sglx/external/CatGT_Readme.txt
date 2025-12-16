Purpose:
+ Optionally join trials with given run_name and g-index in t-index range [ta,tb]...
+ ...Or run on any individual file.
+ Optionally apply bandpass and global demux CAR filters.
+ Optionally edit out saturation artifacts.
+ Optionally extract tables of sync waveform edge times to drive TPrime.
+ Optionally extract tables of any other TTL event times to be aligned with spikes.

Output:
+ Results are placed next to source, named like this, with t-index = 'cat':
+    path/run_name_g5_tcat.imec1.ap.bin.
+ Errors and run messages are appended to CatGT.log in the app folder.

Usage:
>CatGT -dir=data_dir -run=run_name -g=g -t=ta,tb <which streams> [ options ]

Usage notes:
+ It is easiest to run from a .bat file like the included example 'runit.bat'. Edit that file then double-click it to run.
+ Options must not have spaces.
+ In *.bat files, continue long lines using <space><caret>. Like this ^.
+ Remove all white space at line ends, especially after a caret (^).
+ Read CatGT.log. There is no interesting output in the command window.

Which streams:
-ap                  ;required to process ap streams
-lf                  ;required to process lf streams
-ni                  ;required to process ni stream
-prb_3A              ;if -ap or -lf process 3A-style probe files, e.g. run_name_g0_t0.imec.ap.bin
-prb=0,3:5           ;if -ap or -lf AND !prb_3A process these probes

Options:
-no_run_fld          ;older data generated before run folders were automatic
-prb_fld             ;use folder-per-probe organization
-prb_miss_ok         ;instead of stopping, silently skip missing probes
-t=cat               ;extract TTL from CatGT output files (instead of -t=ta,tb)
-exported            ;apply FileViewer 'exported' tag to in/output filenames
-t_miss_ok           ;instead of stopping, zero-fill if trial missing
-aphipass=300        ;apply ap high pass filter (float Hz)
-aplopass=9000       ;apply ap low  pass filter (float Hz)
-lfhipass=0.1        ;apply lf high pass filter (float Hz)
-lflopass=1000       ;apply lf low  pass filter (float Hz)
-loccar=2,8          ;apply ap local CAR annulus (exclude radius,include radius)
-gbldmx              ;apply ap global demux filter
-gfix=0.40,0.10,0.02 ;rmv ap artifacts: ||amp(mV)||, ||slope(mV/sample)||, ||noise(mV)||
-chnexcl=0,3:5       ;exclude these acq chans from ap loccar, gbldmx and gfix
-SY=0,384,6,500      ;extract TTL signal from imec SY (probe,word,bit,millisec)
-XA=2,3.0,0.08,25    ;extract TTL signal from nidq XA (word,thresh(v),min(V),millisec)
-XD=8,0,0            ;extract TTL signal from nidq xD (word,bit,millisec)
-dest=path           ;alternate path for output files (must exist)
-out_prb_fld         ;if using -dest, create output subfolder per probe

Parameter notes:
- The input run_name is a base (undecorated) name without g- or t-indices.
- The input files are expected to be organized as SpikeGLX writes them, that is:
  + If -no_run_fld option: data_dir/run_name_g0_t0.imec0.ap.bin.
  + No -prb_fld option: data_dir/run_name_g0/run_name_g0_t0.imec0.ap.bin.
  + If -prb_fld option: data_dir/run_name_g0/run_name_g0_imec0/run_name_g0_t0.imec0.ap.bin.
- Use option -prb_miss_ok when run output is split across multiple drives.
- Operate on a single t-index like this: -t=ta,ta.
- Operate on CatGT output files (TTL extraction only) like this: -t=cat.
- New .bin/.meta files are output only if trials are concatenated (tb > ta), or if filters are applied.
- The -dest option will create an output subfolder for the run: path/catgt_run_name.

- A meta file is also created, e.g.: path/run_name_g5_tcat.imec1.ap.meta.
- The meta file also gets 'catGTCmdlineN=<command line string>'.
- The meta file gets a tag indicating the actual range of t-indices used: 'catTVals=0,20'.

- loccar option:
  + Do CAR common average referencing on an annular area about each site.
  + Specify an excluded inner radius (in sites) and an outer radius.

- gbldmx option:
  + Also use -aphipass to remove DC offsets.
  + No, filter options -loccar and -gbldmx don't make sense together.

- gfix option:
  + Light or chewing artifacts often make large amplitude excursions on a majority of channels. This tool identifies them and cuts them out, replacing with zeros. You specify three things. (1) A minimum absolute amplitude in mV (zero ignores the amplitude test). (2) A minimum absolute slope in mV/sample (zero ignores the slope test). (3) A noise level in mV defining the end of the transient.
  + Also use -aphipass to remove DC offsets.
  + Use the SpikeGLX FileViewer to select appropriate amplitude and slope values for a given run. Be sure to turn highpass filtering ON and spatial <S> filters OFF to match the conditions the CatGT artifact detector will use. Zoom the time scale (ctrl + click&drag) to see the individual sample points and their connecting segments. Set the slope this way: Zoom in on the artifact initial peak, the points with greatest amplitude. Suppose consecutive points {A,B,C,D} make up the peak and {B,C,D} exceed the amplitude threshold. Then there are three slopes {B-A,C-B,D-C} connecting these points. Select the largest value. That is, set the slope to the fastest voltage change near the peak. An artifact will usually be several times faster than a neuronal spike.
  + Yes, -gbldmx and -gfix make sense used together.

- TTL (digital) extractions:
  + XA are analog-type NI channels. The signal is parametrized by index of the word in the stored data file, the low-to-high TTL threshold (V), minimum required amplitude (V), and milliseconds high.
  + For square pulses, set minimum <= thresh to ignore the minimum parameter and run more efficiently. For non-square pulses set minimum > thresh to ensure pulse attains desired amplitude.
  + SpikeGLX MA channels can also be scanned with the -XA option.
  + SY (imec), XD (NI) are digital-type channels. The signal is parametrized by index of word in the stored data file, index of the bit in the word, and milliseconds high.
  + All indexing is zero-based.
  + Milliseconds high means the signal must persist above threshold for that long.
  + Milliseconds high can be zero to specify detection of all rising edges regardless of pulse duration.
  + Milliseconds high default precision (tolerance) is +/- 20%.
    * Default tolerance can be overriden by appending it in milliseconds as the last parameter for that extractor.
    * Each extractor can have its own tolerance.
    * E.g. -XD=8,0,100   seeks pulses with duration in default range [80,120] ms.
    * E.g. -XD=8,0,100,2 seeks pulses with duration in specified range [98,102] ms.
  + A given bit could encode two or more types of pulse that have different durations, E.g...
  + -XD=8,0,10 -XD=8,0,20 scans and reports both 10 and 20 ms pulses on the same line.
  + Each option, say -SY=0,384,6,500, creates output run_name_g0_tcat.imec0.SY_384_6_500.txt.
  + The threshold is not encoded in the -XA filename; just word and millisec.
  + The files report the times (s) of leading edges of detected pulses; one time per line, <\n> line endings.
  + The time is relative to the start of the stream in which the pulse is detected (native time).
  + Option -t=cat allows you to concatenate/filter the data in a first pass and later extract TTL events from the output files which are now named 'tcat'. NOTE: If the files to operate on are now in an output folder named 'catgt_run_name' then DO PUT 'catgt_' in the -run parameter like example (2):
    * Example (1) Saving to native folders --
        Pass 1: >CatGT -dir=aaa -run=bbb -t=ta,tb.
        Pass 2: >CatGT -dir=aaa -run=bbb -t=cat.
    * Example (2) Saving to dest folders --
        Pass 1: >CatGT -dir=aaa -run=bbb -t=ta,tb -dest=ccc.
        Pass 2: >CatGT -dir=ccc -run=catgt_bbb -t=cat -dest=ccc.
    * NOTE: Second pass is restricted to TTL extraction. An error is flagged if the second pass specifies any concatenation or filter options.