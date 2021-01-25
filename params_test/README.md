# Compute derived-EMG for multiple parameters and compare with "ground truth"


### Usage:

Set constants in `run_EMG_tests.py` and run it in a venv with EMGfromLFP installed. derived EMGs will be computed for all combinations of
parameters. All data will be saved in DATA_DIR.

Then run `analyze_EMG_tests.py`.
It will compute the RMS of the true EMG ( The source data must have an EMG
channel (the name of which is to be set in EMG_CHANNAME) ), compute and display
the correlation with all the derived EMGs, and plot the best fitting derived EMG
along with the RMS of the true EMG for comparison. All results are saved in
`RESULTS_DIR`
