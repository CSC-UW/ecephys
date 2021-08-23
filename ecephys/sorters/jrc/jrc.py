from pathlib import Path
import os
import matlab.engine
from ...sglx import utils as sglx_utils
from ...sglx.external import SGLXMetaToCoords

"""Run JRClust."""

JRCLUST_PATH = '/Volumes/scratch/neuropixels/matlab/external/JRCLUST-4.0.0'

JRC_PARAMS_DF = {
    'CARMode': 'mean',  # average reference?
    'refracInt': '0.25',  # average reference?
}


def run_JRC(binpath, jrc_output_dir, jrc_params=None, badChans=None,
            jrclust_path=JRCLUST_PATH, config_name='config'):
    """Generate `config.prm` and run `jrc detect config.prm` for bin file.

    Args:
        binpath (str or path-like): path to SGLX bin
        jrc_output_dir: output directory

    Kwargs:
        jrc_params (dict): unused (TODO)
        badChans (list): List of channels to ignore (1-indexed)
    """

    if jrc_params is None:
        jrc_params = {}
    binpath = Path(binpath)

    config_str = get_jrc_config_str(binpath, jrc_output_dir,
                                    jrc_params=jrc_params, badChans=badChans)
    config_file = f'{config_name}.prm'

    with open(Path(jrc_output_dir)/config_file, 'w') as f:
        f.write(config_str)

    print(
        f'Running jrc detect for `{binpath}`. Results at `{jrc_output_dir}`.'
    )
    print(
        f'JRC config: {jrc_params}'
    )

    module_dir_path = os.path.dirname(os.path.realpath(__file__))

    eng = matlab.engine.start_matlab()
    eng.addpath(eng.genpath(module_dir_path))  # `jrc_detect.m` on path
    eng.addpath(eng.genpath(jrclust_path))
    eng.cd(str(jrc_output_dir))  # Save output there
    eng.pwd()
    eng.jrc_detect(config_file, nargout=0)
    # eng.run('jrc_detect.m', nargout=0)

    print(
        f'Done running jrc detect for `{binpath}`. Results at `{jrc_output_dir}`'
    )


def get_jrc_config_str(binpath, jrc_output_dir, jrc_params=None, badChans=None):
    """Bootstrap binpath to generate config file."""

    if jrc_params is None:
        jrc_params = {}
    if badChans is None:
        badChans = []

    binpath = Path(binpath)
    jrc_output_dir = Path(jrc_output_dir)
    # Incorporate Ephys params
    _, sRate, nChans, uVperBit = sglx_utils.EphysParams(binpath)
    # Incorporate site locations information
    metaName = binpath.stem + ".meta"
    metaPath = Path(binpath.parent / metaName)
    SGLXMetaToCoords.MetaToCoords(
        metaPath, 2, destFullPath=str(jrc_output_dir/'jrc_probe_str.txt')
    )  # 2 -> JRC config' probe string
    with open(jrc_output_dir/'jrc_probe_str.txt') as f:
        prb_str = f.read()
    # Incorporate other params
    for key in jrc_params:
        if key not in JRC_PARAMS_DF:
            raise ValueError('Unrecognized key')
    params = dict(JRC_PARAMS_DF)
    params.update(jrc_params)
    params = {k: str(v) for k, v in params.items()}

    cfg_str = """
% JRCLUST parameters (common parameters only)
% For a description of these parameters, including legal options, see https://jrclust.readthedocs.io/en/latest/parameters/index.html

% USAGE PARAMETERS
outputDir = ''; % Directory in which to place output files (Will output to the same directory as this file if empty)

% PROBE PARAMETERS
probePad = [12, 12]; % (formerly vrSiteHW) Recording contact pad size (in μm) (Height x width)
""" + prb_str + """

% RECORDING FILE PARAMETERS
bitScaling = """ + str(uVperBit) + """; % (formerly uV_per_bit) ADC bit scaling factor (Conversion factor for ADC bit values to μV)
nChans = """ + str(nChans) + """; % Number of channels stored in recording file (Distinct from the number of AP sites)
rawRecordings = {'""" + str(os.path.realpath(binpath)) + """'}; % Path or paths to raw recordings to sort
sampleRate = """ + str(sRate) + """; % (formerly sRateHz) Sampling rate (in Hz) of raw recording

% PREPROCESSING PARAMETERS
freqLimBP = [300, 3000]; % (formerly freqLim) Frequency cutoffs for bandpass filter
ignoreSites = """ + str(badChans) + """; % (formerly viSiteZero) Site IDs to ignore manually

% SPIKE DETECTION PARAMETERS
CARMode = '""" + params['CARMode'] + """';
refracInt = """ + params['refracInt'] + """;
evtWindow = [-0.25, 0.75]; % (formerly spkLim_ms) Time range (in ms) of filtered spike waveforms, centered at the peak
nSiteDir = 4.5; % (formerly maxSite) Number of neighboring sites to group in either direction (nSitesEvt is set to 1 + 2*nSiteDir - nSitesExcl)
nSitesExcl = 6; % (formerly nSites_ref) Number of sites to exclude from the spike waveform group for feature extraction
evtWindowRaw = [-0.5, 1.5]; % (formerly spkLim_raw_ms) Time range (in ms) of raw spike waveforms, centered at the peak


% DISPLAY PARAMETERS
dispTimeLimits = [0, 0.2]; % (formerly tlim) Time range (in ms) to display
colorMap = [0.5, 0.5, 0.5; 0, 0, 0; 1, 0, 0]; % (formerly mrColor_proj) RGB color map for background, primary selected, and secondary selected spikes (The first three values are the R values, the next three are the G values, and the last three are the B values.)
corrRange = [0.9, 1]; % (formerly corrLim) Correlation score range to distinguish by color map
pcPair = [1, 2]; % Pair of PCs to display
"""

    return cfg_str
