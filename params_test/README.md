# Compute derived-EMG for multiple parameters and compare with "ground truth"


### Usage:

Set constants in `run_EMG_tests.py` and run it in a venv with sleepscore and EMGfromLFP installed. derived EMGs will be computed for all combinations of parameters.

Then run `analyze_EMG_tests.py`.
It will compute the RMS of the true EMG ( The source data must have an EMG
channel (assumed to be called "EMGs-1") ) and compute the correlation with all
the derived EMGs.
