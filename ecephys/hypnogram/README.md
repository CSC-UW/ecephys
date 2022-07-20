## HypnoGra(ha)m

### Core classes
This package provides a Hypnogram class, which is just a subclassed Pandas DataFrame with some useful methods. Every hypnogram should have `state`, `start_time`, `end_time`, and `duration` columns, and each row should be one bout of a vigilance state. For example:
```
state 	start_time 	end_time 	duration
NREM 	2020-09-24 15:19:34.989999771 	2020-09-24 15:19:42.989999771 	0 days 00:00:08
```
- DatetimeHypnogram objects (e.g. above) expect all start and end times to be pandas datetime objects, and all durations to be pandas timedelta objects.
- FloatHypnograms expect all time/duration fields to contain floats, which represent times in seconds. FloatHypnogram.as_datetime() can be used to easily convert a FloatHypnogram, which is the natural output format of many sleep scoring softwares, to a DatetimeHypnogram format, which is much easier to manipulate and work with.

### Dependencies
`["numpy", "pandas"]`